# ============================================================
# PART 1: Data Schema and Structure (UNCHANGED)
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import requests
from bs4 import BeautifulSoup
import re
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgeGroup(Enum):
    INFANT = "0-12 months"
    TODDLER = "1-3 years"
    PRESCHOOL = "3-5 years"
    SCHOOL_AGE = "5-12 years"
    TEEN = "12-18 years"
    ALL_AGES = "All ages (with restrictions)"

class IllnessCategory(Enum):
    RESPIRATORY = "respiratory"
    DIGESTIVE = "digestive"
    FEVER = "fever"
    SKIN = "skin"
    PAIN = "pain"
    SLEEP = "sleep"
    ALLERGY = "allergy"
    OTHER = "other"

@dataclass
class Remedy:
    """Structured remedy with all necessary metadata"""
    title: str
    illness_category: IllnessCategory
    specific_symptoms: List[str]
    ingredients: List[str]
    preparation_steps: List[str]
    dosage: Optional[str] = None
    age_min_months: int = 0
    age_max_months: Optional[int] = None
    contains_honey: bool = False
    contraindications: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    description: str = ""
    why_it_works: str = ""
    duration: Optional[str] = None
    source_url: str = ""
    trust_score: str = "medium"
    evidence_level: str = "traditional"
    
    def to_dict(self) -> Dict:
        return {
            "type": "food_remedy",
            "title": self.title,
            "illness_category": self.illness_category.value,
            "specific_symptoms": self.specific_symptoms,
            "ingredients": self.ingredients,
            "steps": self.preparation_steps,
            "dosage": self.dosage,
            "age_min_months": self.age_min_months,
            "age_max_months": self.age_max_months,
            "contains_honey": self.contains_honey,
            "contraindications": self.contraindications,
            "warnings": self.warnings,
            "body": self.description,
            "why_it_works": self.why_it_works,
            "duration": self.duration,
            "source_url": self.source_url,
            "trust_score": self.trust_score,
            "evidence_level": self.evidence_level
        }
    
    def to_text(self) -> str:
        symptoms_text = ", ".join(self.specific_symptoms)
        ingredients_text = ", ".join(self.ingredients)
        steps_text = " ".join(self.preparation_steps)
        
        return f"""
        Remedy: {self.title}
        Category: {self.illness_category.value}
        Symptoms: {symptoms_text}
        Description: {self.description}
        Why it works: {self.why_it_works}
        Ingredients: {ingredients_text}
        Preparation: {steps_text}
        Safe for: {self.age_min_months}+ months
        """


# ============================================================
# PART 2: Web Scraper with FIXED Parsing
# ============================================================

class RemedyParser:
    """Parse web pages and extract structured remedies"""
    
    SYMPTOM_MAPPING = {
        "cough": IllnessCategory.RESPIRATORY,
        "cold": IllnessCategory.RESPIRATORY,
        "sore throat": IllnessCategory.RESPIRATORY,
        "runny nose": IllnessCategory.RESPIRATORY,
        "congestion": IllnessCategory.RESPIRATORY,
        "stuffy nose": IllnessCategory.RESPIRATORY,
        "fever": IllnessCategory.FEVER,
        "stomach": IllnessCategory.DIGESTIVE,
        "belly": IllnessCategory.DIGESTIVE,
        "bellyache": IllnessCategory.DIGESTIVE,
        "constipation": IllnessCategory.DIGESTIVE,
        "diarrhea": IllnessCategory.DIGESTIVE,
        "bloating": IllnessCategory.DIGESTIVE,
        "indigestion": IllnessCategory.DIGESTIVE,
        "headache": IllnessCategory.PAIN,
        "earache": IllnessCategory.PAIN,
        "rash": IllnessCategory.SKIN,
        "acne": IllnessCategory.SKIN,
        "dry skin": IllnessCategory.SKIN,
        "sunburn": IllnessCategory.SKIN,
        "allergy": IllnessCategory.ALLERGY,
        "pain": IllnessCategory.PAIN,
    }
    
    KEY_INGREDIENTS = [
        "honey", "lemon", "salt", "water", "tea", "chamomile", "peppermint",
        "ginger", "garlic", "apple cider vinegar", "eucalyptus", "chicken soup",
        "saline", "zinc", "vitamin c", "cayenne pepper", "menthol", "lavender",
        "sage", "vapor rub", "echinacea", "elderberry"
    ]
    
    @staticmethod
    def extract_age_restriction(text: str) -> int:
        """Extract minimum age in months from text"""
        text_lower = text.lower()
        
        # Check for explicit "under 1 year" or "younger than 1" restrictions
        if any(phrase in text_lower for phrase in [
            "younger than 1", "under 1 year", "never give honey",
            "not for children under 1", "under age 1"
        ]):
            return 12
        
        # Look for year restrictions
        year_match = re.search(r'(\d+)\s*year[s]?\s*or\s*older', text, re.IGNORECASE)
        if year_match:
            return int(year_match.group(1)) * 12
        
        year_match = re.search(r'(\d+)\s*year[s]?', text, re.IGNORECASE)
        if year_match:
            years = int(year_match.group(1))
            # Check context
            pre_text = text[:year_match.start()].lower()
            if any(word in pre_text for word in ['under', 'less than', 'younger than']):
                return years * 12
            # If it says "5 years or older", return that age
            post_text = text[year_match.end():year_match.end()+20].lower()
            if 'older' in post_text or 'and up' in post_text:
                return years * 12
        
        # Look for month restrictions
        month_match = re.search(r'(\d+)\s*month[s]?', text, re.IGNORECASE)
        if month_match:
            return int(month_match.group(1))
        
        return 0
    
    @staticmethod
    def contains_honey(ingredients: List[str]) -> bool:
        """Check if remedy contains honey"""
        return any("honey" in ing.lower() for ing in ingredients)
    
    @staticmethod
    def categorize_symptom(text: str) -> IllnessCategory:
        """Determine category from symptom text"""
        text_lower = text.lower()
        for symptom, category in RemedyParser.SYMPTOM_MAPPING.items():
            if symptom in text_lower:
                return category
        return IllnessCategory.OTHER
    
    @classmethod
    def parse_chkd_content(cls, html: str, url: str) -> List[Remedy]:
        """Parse CHKD blog structure - COMPLETELY REWRITTEN"""
        soup = BeautifulSoup(html, 'html.parser')
        remedies = []
        
        # The CHKD page has a simple structure: "For a X" followed by bullet points
        # Find all text content
        text_content = soup.get_text()
        
        # Split by the "For a" pattern - fixed regex
        sections = re.split(r'(For a [^\n]+)', text_content)
        
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
                
            header = sections[i].strip()
            content = sections[i + 1].strip()
            
            # Extract symptom from header
            symptom = header.replace("For a ", "").replace("For ", "").strip()
            category = cls.categorize_symptom(symptom)
            
            # Split content into bullet points/sentences
            bullets = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('Call your doctor')]
            
            if not bullets:
                continue
            
            # Process each bullet as a potential remedy
            combined_content = " ".join(bullets[:5])  # Take first 5 bullets for context
            
            remedy = cls._parse_remedy_text(
                combined_content,
                symptom,
                category,
                url,
                source_trust="high"
            )
            
            if remedy:
                remedies.append(remedy)
                logger.info(f"✓ Extracted: {remedy.title}")
        
        return remedies
    
    @classmethod
    def parse_healthline_content(cls, html: str, url: str) -> List[Remedy]:
        """Parse Healthline remedies - they're numbered sections with descriptions"""
        soup = BeautifulSoup(html, 'html.parser')
        remedies = []
        
        # Extract all text from article
        main_content = soup.find('div', class_=lambda x: x and 'article' in x.lower())
        if not main_content:
            main_content = soup.find('main') or soup
        
        text = main_content.get_text(separator='\n')
        
        # Split by numbered remedies: "1. ", "2. ", etc.
        # Pattern: digit(s) + dot + remedy name
        remedy_pattern = r'^(\d+)\.\s+([^\n]+)'
        
        sections = re.split(r'\n(?=\d+\.\s+)', text)
        
        logger.info(f"Found {len(sections)} potential remedy sections")
        
        for section in sections:
            lines = section.strip().split('\n')
            if not lines or not lines[0]:
                continue
            
            # Extract remedy number and title
            first_line = lines[0]
            match = re.match(r'(\d+)\.\s+(.+)', first_line)
            
            if not match:
                continue
            
            remedy_number = match.group(1)
            remedy_title = match.group(2).strip()
            
            # Combine remaining lines as content
            content_lines = lines[1:]
            full_content = ' '.join(line.strip() for line in content_lines if line.strip())
            
            if not full_content or len(full_content) < 30:
                logger.debug(f"⊘ Remedy {remedy_number} has insufficient content")
                continue
            
            logger.info(f"📍 Processing remedy {remedy_number}: {remedy_title}")
            logger.info(f"   Content: {full_content[:100]}...")
            
            # Parse the remedy
            remedy = cls._parse_remedy_text(
                full_content,
                "Cold/Flu",
                IllnessCategory.RESPIRATORY,
                url,
                title_override=remedy_title,
                source_trust="medium",
                evidence="anecdotal"
            )
            
            if remedy:
                # Check for research mentions
                if any(word in full_content.lower() for word in [
                    "study", "research", "evidence", "clinical", "shown"
                ]):
                    remedy.evidence_level = "research-backed"
                
                remedies.append(remedy)
                logger.info(f"  ✅ Created: {remedy.title}")
            else:
                logger.debug(f"  ⊘ Failed to parse")
        
        logger.info(f"✓ Total remedies extracted: {len(remedies)}")
        return remedies

    
    @classmethod
    def _parse_remedy_text(cls, text: str, symptom: str, category: IllnessCategory,
                          url: str, title_override: Optional[str] = None,
                          source_trust: str = "medium", evidence: str = "traditional") -> Optional[Remedy]:
        """Parse individual remedy from text - IMPROVED EXTRACTION"""
        
        # 1. Extract ingredients
        ingredients = []
        text_lower = text.lower()
        
        for key_ing in cls.KEY_INGREDIENTS:
            if key_ing in text_lower:
                # Try to extract quantity if present
                pattern = rf'(\d+[\s-]*(tablespoon|teaspoon|tsp|tbsp|cup|drop|ml)?s?\s+(?:of\s+)?{re.escape(key_ing)})'
                match = re.search(pattern, text_lower)
                if match:
                    ingredients.append(match.group(1))
                elif key_ing not in [ing.lower() for ing in ingredients]:
                    ingredients.append(key_ing)
        
        # Check for action-based remedies (non-ingredient)
        is_action_remedy = any(action in text_lower for action in [
            "gargle", "massage", "inhale", "steam", "humidifier",
            "rest", "sleep", "wash", "compress"
        ])
        
        if not ingredients and not is_action_remedy:
            return None
        
        # 2. Extract preparation steps
        steps = []
        sentences = re.split(r'[.!?]+', text)
        
        action_verbs = [
            "mix", "add", "give", "apply", "massage", "drink", "gargle",
            "sip", "steep", "inhale", "run", "take", "put", "make",
            "boil", "heat", "cool", "serve", "use", "place", "wrap"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Check if sentence contains action verb
            if any(verb in sentence.lower() for verb in action_verbs):
                steps.append(sentence)
        
        # If no steps found, use first few sentences as description
        if not steps and text:
            steps = [s.strip() for s in sentences[:3] if s.strip()]
        
        # 3. Safety and age restrictions
        age_min = cls.extract_age_restriction(text)
        contains_honey_flag = cls.contains_honey(ingredients)
        
        if contains_honey_flag and age_min < 12:
            age_min = 12
        
        # 4. Create title
        if title_override:
            title = title_override.strip()
        elif ingredients:
            main_ingredient = ingredients[0].title()
            title = f"{main_ingredient} for {symptom.title()}"
        else:
            # Action-based remedy
            action_match = re.search(r'(steam|gargle|massage|compress|humidifier)', text_lower)
            if action_match:
                title = f"{action_match.group(1).title()} Therapy for {symptom.title()}"
            else:
                title = f"Natural Remedy for {symptom.title()}"
        
        # 5. Extract warnings and contraindications
        contraindications = []
        warnings = []
        
        for sentence in sentences:
            s_lower = sentence.lower().strip()
            if any(phrase in s_lower for phrase in [
                "not for", "never give", "avoid if", "do not give",
                "contraindication", "should not"
            ]):
                contraindications.append(sentence.strip())
            
            if any(phrase in s_lower for phrase in [
                "warning", "call", "doctor", "seek medical",
                "side effect", "caution", "consult"
            ]):
                warnings.append(sentence.strip())
        
        # 6. Extract dosage if present
        dosage = None
        dosage_match = re.search(
            r'(\d+[\s-]*(tablespoon|teaspoon|tsp|tbsp|cup|drop)[s]?.*?(?:daily|per day|times|hourly))',
            text,
            re.IGNORECASE
        )
        if dosage_match:
            dosage = dosage_match.group(1)
        
        # 7. Extract "why it works" information
        why_works = ""
        for sentence in sentences:
            if any(word in sentence.lower() for word in [
                "helps", "soothes", "relieves", "contains", "properties",
                "works by", "effective because", "anti-inflammatory", "antibacterial"
            ]):
                why_works = sentence.strip()
                break
        
        return Remedy(
            title=title,
            illness_category=category,
            specific_symptoms=[symptom.lower()],
            ingredients=ingredients,
            preparation_steps=steps[:5],  # Limit to 5 steps
            dosage=dosage,
            age_min_months=age_min,
            contains_honey=contains_honey_flag,
            description=text[:500],  # First 500 chars
            why_it_works=why_works,
            source_url=url,
            trust_score=source_trust,
            evidence_level=evidence,
            contraindications=contraindications,
            warnings=warnings
        )


# ============================================================
# PART 3: FAISS Database Builder (UNCHANGED)
# ============================================================

class RemedyDatabaseBuilder:
    """Build FAISS database with structured remedies"""
    
    def __init__(self, faiss_path: str = "faiss_indices/food_remedies"):
        self.faiss_path = Path(faiss_path)
        self.faiss_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            logger.warning(f"OpenAIEmbeddings failed: {e}")
            self.embeddings = None
        
        self.remedies: List[Remedy] = []
    
    def add_remedy(self, remedy: Remedy):
        """Add a structured remedy to the collection"""
        self.remedies.append(remedy)
        logger.info(f"Added remedy: {remedy.title}")
    
    def scrape_and_parse(self, urls: List[str]):
        """Scrape URLs and parse remedies"""
        parser = RemedyParser()
        
        for url in urls:
            try:
                logger.info(f"Scraping {url}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, timeout=15, headers=headers)
                response.raise_for_status()
                
                remedies = []
                
                if "chkd.org" in url:
                    remedies = parser.parse_chkd_content(response.text, url)
                elif "healthline.com" in url:
                    remedies = parser.parse_healthline_content(response.text, url)
                else:
                    logger.warning(f"No parser for {url}")
                    continue
                
                for remedy in remedies:
                    self.add_remedy(remedy)
                
                logger.info(f"✅ Scraped {len(remedies)} remedies from {url}")
                
            except Exception as e:
                logger.error(f"❌ Error scraping {url}: {e}")
    
    def build_faiss_index(self):
        """Build and save FAISS index"""
        if not self.remedies:
            logger.error("No remedies to index!")
            return
        
        if not self.embeddings:
            logger.error("Cannot build FAISS: Embeddings not initialized")
            return
        
        documents = []
        for remedy in self.remedies:
            doc = Document(
                page_content=remedy.to_text(),
                metadata=remedy.to_dict()
            )
            documents.append(doc)
        
        logger.info(f"Building FAISS index with {len(documents)} documents...")
        
        try:
            vectordb = FAISS.from_documents(documents, self.embeddings)
            vectordb.save_local(str(self.faiss_path))
            logger.info(f"✅ FAISS index saved to {self.faiss_path}")
        except Exception as e:
            logger.error(f"❌ Failed to build FAISS: {e}")
    
    def add_manual_remedies(self):
        """Add well-structured remedies manually"""
        
        # Honey and Lemon for Cough
        self.add_remedy(Remedy(
            title="Honey and Lemon for Cough",
            illness_category=IllnessCategory.RESPIRATORY,
            specific_symptoms=["cough", "sore throat"],
            ingredients=["1 tablespoon raw honey", "1 tablespoon fresh lemon juice", "warm water (optional)"],
            preparation_steps=[
                "Mix 1 tablespoon of raw honey with 1 tablespoon fresh lemon juice",
                "Can be taken directly or mixed with warm water",
                "Give to child 2-3 times per day as needed"
            ],
            dosage="1 tablespoon mixture, 2-3 times daily",
            age_min_months=12,
            contains_honey=True,
            contraindications=["Never give to infants under 12 months"],
            warnings=["May cause infant botulism in babies under 1 year"],
            description="Honey coats and soothes the throat while lemon provides vitamin C and helps break down mucus.",
            why_it_works="Honey has natural antibacterial properties and coats irritated throat tissues. Studies show it's as effective as dextromethorphan for nighttime cough.",
            duration="Use as needed for 3-7 days",
            source_url="https://www.chkd.org/patient-family-resources/our-blog/home-remedies-for-sick-kids/",
            trust_score="high",
            evidence_level="research-backed"
        ))
        
        # Chamomile Tea
        self.add_remedy(Remedy(
            title="Chamomile Tea for Upset Stomach",
            illness_category=IllnessCategory.DIGESTIVE,
            specific_symptoms=["stomach ache", "upset stomach", "indigestion", "belly pain"],
            ingredients=["1 chamomile tea bag or 1 tsp dried chamomile", "1 cup hot water", "honey (optional, for age 1+)"],
            preparation_steps=[
                "Steep chamomile tea bag in hot water for 5-10 minutes",
                "Let cool to comfortable drinking temperature",
                "Add honey if desired (only for children 1 year and older)",
                "Have child sip slowly"
            ],
            dosage="1 cup, 2-3 times daily as needed",
            age_min_months=6,
            contains_honey=False,
            description="Chamomile has natural anti-inflammatory and calming properties that soothe digestive discomfort.",
            why_it_works="Chamomile contains compounds that relax digestive muscles and reduce inflammation in the GI tract.",
            duration="Use as needed during stomach upset",
            source_url="https://www.chkd.org/patient-family-resources/our-blog/home-remedies-for-sick-kids/",
            trust_score="high",
            evidence_level="traditional"
        ))
        
        # Steam Inhalation
        self.add_remedy(Remedy(
            title="Steam Inhalation for Nasal Congestion",
            illness_category=IllnessCategory.RESPIRATORY,
            specific_symptoms=["congestion", "stuffy nose", "blocked nose", "chest congestion"],
            ingredients=["Hot water", "Large bowl or use bathroom shower", "Optional: 2-3 drops eucalyptus oil"],
            preparation_steps=[
                "Method 1: Run hot shower to create steam, sit in bathroom with door closed for 10-15 minutes",
                "Method 2: Pour hot water in large bowl, have child lean over with towel over head (supervise closely)",
                "If using eucalyptus oil, add 2-3 drops to water",
                "Have child breathe deeply through nose",
                "Do not let water get too hot to prevent burns"
            ],
            dosage="10-15 minutes, 2-3 times daily",
            age_min_months=24,
            contraindications=["Not for infants under 2 years without supervision"],
            warnings=["Always supervise to prevent burns", "Keep face at safe distance from hot water"],
            description="Steam helps loosen mucus and opens nasal passages for easier breathing.",
            why_it_works="Warm, moist air helps thin mucus secretions and reduces inflammation in nasal passages.",
            duration="Use during cold or congestion episodes",
            source_url="https://www.chkd.org/patient-family-resources/our-blog/home-remedies-for-sick-kids/",
            trust_score="high",
            evidence_level="traditional"
        ))


# ============================================================
# PART 4: Usage Example
# ============================================================

if __name__ == "__main__":
    builder = RemedyDatabaseBuilder()
    
    SEED_URLS = [
        "https://www.chkd.org/patient-family-resources/our-blog/home-remedies-for-sick-kids/",
        "https://www.healthline.com/health/cold-flu/home-remedies",
    ]
    
    print("=" * 60)
    print("REMEDY DATABASE BUILDER")
    print("=" * 60)
    
    # 1. Add manual remedies
    print("\n📝 Adding manual remedies...")
    builder.add_manual_remedies()
    
    # 2. Scrape web sources
    print("\n🌐 Scraping web sources...")
    builder.scrape_and_parse(SEED_URLS)
    
    # 3. Build the index
    print("\n🔨 Building FAISS index...")
    builder.build_faiss_index()
    
    # 4. Summary
    print("\n" + "=" * 60)
    print(f"✅ Built database with {len(builder.remedies)} remedies")
    print("=" * 60)
    
    print("\n📊 Categories covered:")
    categories = {}
    for remedy in builder.remedies:
        cat = remedy.illness_category.value
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count} remedies")
    
    print("\n📋 Sample remedies:")
    for i, remedy in enumerate(builder.remedies[:5], 1):
        print(f"  {i}. {remedy.title}")
        print(f"     Category: {remedy.illness_category.value}")
        print(f"     Ingredients: {', '.join(remedy.ingredients[:3])}")
        print(f"     Age: {remedy.age_min_months}+ months")
        print()
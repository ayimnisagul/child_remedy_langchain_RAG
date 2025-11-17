"""Display formatting utilities."""

import streamlit as st
from typing import Dict, List


def render_remedy(remedy: Dict) -> None:
    """Pretty, structured rendering of one remedy with numbered steps."""
    st.markdown(f"### 🌿 {remedy.get('title', 'Untitled Remedy')}")
    
    badges = []
    if remedy.get("contains_honey"):
        badges.append("🍯 Contains honey")
    if remedy.get("trust_score") == "high":
        badges.append("✅ High trust")
    if remedy.get("evidence_level") == "research-backed":
        badges.append("🔬 Research-backed")
    
    if badges:
        st.caption(" • ".join(badges))
    
    st.caption(f"👶 Safe for: {remedy.get('age_min_months', 0)}+ months")
    st.markdown("---")

    if remedy.get("description"):
        st.markdown("**What it does:**")
        st.write(remedy["description"])

    if remedy.get("why_it_works"):
        with st.expander("🔬 Why it works", expanded=False):
            st.write(remedy["why_it_works"])

    if remedy.get("ingredients"):
        st.markdown("**🥄 Ingredients:**")
        for ing in remedy["ingredients"]:
            st.markdown(f"- {ing}")

    if remedy.get("steps"):
        st.markdown("**🧾 Preparation Steps:**")
        for i, step in enumerate(remedy["steps"], 1):
            st.markdown(f"{i}. {step}")

    if remedy.get("dosage"):
        st.markdown(f"**👧 Dosage:** {remedy['dosage']}")
    if remedy.get("duration"):
        st.markdown(f"**⏳ Duration:** {remedy['duration']}")

    warnings = remedy.get("warnings", []) + remedy.get("contraindications", [])
    if warnings:
        st.markdown("**⚠️ Safety Information:**")
        for w in warnings:
            st.warning(w)

    if remedy.get("source_url"):
        st.caption(f"📚 Source: [{remedy['source_url']}]({remedy['source_url']})")
    
    st.markdown("---")


def render_remedies_list(remedies: List[Dict]) -> None:
    """Render multiple remedies in expanders."""
    for i, remedy in enumerate(remedies):
        with st.expander(f"🌿 {remedy['title']}", expanded=(i == 0)):
            render_remedy(remedy)


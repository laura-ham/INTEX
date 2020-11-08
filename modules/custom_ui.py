import streamlit as st

def sidebar_settings():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    max_width = 1000
    max_width_100_percent = False
    _set_block_container_style(max_width, max_width_100_percent)
    _set_sidebar_content_style()

def _set_block_container_style(max_width: int = 800, max_width_100_percent: bool = False):
    if max_width_100_percent:
        max_width_str = f"max-width: 95%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
</style>
""",
        unsafe_allow_html=True,
    )

def _set_sidebar_content_style():
    max_width_str = f"width: 30rem;"
    max_width_str_margin = f"margin-left: -30rem;"
    st.markdown(
        f"""
<style>
    .sidebar.--collapsed .sidebar-content{{
        {max_width_str_margin}
    }}
    .sidebar .sidebar-content{{
        {max_width_str}
    }}
</style>
""",
        unsafe_allow_html=True,
    )
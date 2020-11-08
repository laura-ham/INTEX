import streamlit as st
import pyLDAvis
import base64

def get_lda_vis(clustering_pipeline):
    """generates topic-term 2D visualization using pyLDAvis

    Parameters
    ----------
    clustering_pipeline : class reference
        The current modeling pipeling
    """

    with st.spinner(
            "Loading visualization... Once ready, save the generated HTML file shown below."
    ):
        ldavis = clustering_pipeline.generate_ldavis()
        
        # tmp fix for LDAvis error https://stackoverflow.com/questions/47998685/pyldavis-validation-error-on-trying-to-visualize-topics. 
        # not able to comment out line 375 of _prepare pyLDAvis    _input_validate(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency) in dockerized version
        if ldavis == "This visualization is currently not available.":
            st.warning(ldavis)
            return
        
        st.markdown("Set view to widescreen or open HTML in new tab for the best experience.")

        ldavis_html = pyLDAvis.prepared_data_to_html(ldavis)
        b64 = base64.b64encode(ldavis_html.encode()).decode(
        )  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:text/html;base64,{b64}">Download HTML File</a> \
            (right-click and save as &lt;some_name&gt;.html)'

        st.markdown(href, unsafe_allow_html=True)

        iframe = f'<iframe width="100%" height="900" src="data:text/html;base64,{b64}">The “iframe” tag is not supported by your browser.</iframe>'

        st.write(iframe, unsafe_allow_html=True)
        
def get_doctop_vis(clustering_pipeline, media='videos'):
    """ calls a function to generate bokeh visualization

    Parameters
    ----------
    clustering_pipeline : class reference
        The current modeling pipeling
    media : str, optional
        'articles' or 'videos', by default 'videos'
    """
    bokeh_layout = clustering_pipeline.generate_bokeh_umap(media)
    st.bokeh_chart(bokeh_layout)
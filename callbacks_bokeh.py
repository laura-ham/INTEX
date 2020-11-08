from bokeh.models import CustomJS

# handle the currently selected article
def selected_code_videos():
    code = """
            var ids = [];
            var titles = [];
            var published_times = [];
            // var texts = [];
            var publication_end_times = [];
            var media_availables = [];
            var duration_minutes = [];
            var finnpanel_genres = [];
            var links = [];
            cb_data.source.selected.indices.forEach(index => ids.push(source.data['ids'][index]));
            cb_data.source.selected.indices.forEach(index => titles.push(source.data['titles'][index]));
            cb_data.source.selected.indices.forEach(index => published_times.push(source.data['published_times'][index]));
            // cb_data.source.selected.indices.forEach(index => texts.push(source.data['texts'][index]));
            cb_data.source.selected.indices.forEach(index => publication_end_times.push(source.data['publication_end_times'][index]));
            cb_data.source.selected.indices.forEach(index => media_availables.push(source.data['media_availables'][index]));
            cb_data.source.selected.indices.forEach(index => duration_minutes.push(source.data['duration_minutes'][index]));
            cb_data.source.selected.indices.forEach(index => finnpanel_genres.push(source.data['finnpanel_genres'][index]));
            cb_data.source.selected.indices.forEach(index => links.push(source.data['links'][index]));
            var id_title = "<h4>" + ids[0].toString().replace(/<br>/g, ' ') + ': ' + titles[0].toString().replace(/<br>/g, ' ') + "</h4>";
            var publication = "<p1><b>Areena availability: </b> " + published_times[0].toString().replace(/<br>/g, ' ') + ' until ' + publication_end_times[0].toString().replace(/<br>/g, ' ') + "<br>";
            // var full_text = "<h4>" + texts[0].toString().replace(/<br>/g, ' ') + "</h4>";
            var duration_minute = "<p1><b>Duration (minutes): </b> " + duration_minutes[0].toString().replace(/<br>/g, ' ') + "<br>"
            var finnpanel_genres = "<p1><b>Finnpanel genres: </b> " + finnpanel_genres[0].toString().replace(/<br>/g, ' ') + "<br>"
            var link = "<b>Link: </b> <a href='" + links[0].toString() + "' target='_blank'>" + links[0].toString() + "</a></p1>"
            current_selection.text = id_title + publication + duration_minute + finnpanel_genres + link
            current_selection.change.emit();
    """
    return code

def selected_code_articles():
    code = """
            var ids = [];
            var titles = [];
            var published_times = [];
            // var texts = [];
            var links = [];
            cb_data.source.selected.indices.forEach(index => ids.push(source.data['ids'][index]));
            cb_data.source.selected.indices.forEach(index => titles.push(source.data['titles'][index]));
            cb_data.source.selected.indices.forEach(index => published_times.push(source.data['published_times'][index]));
            // cb_data.source.selected.indices.forEach(index => texts.push(source.data['texts'][index]));
            cb_data.source.selected.indices.forEach(index => links.push(source.data['links'][index]));
            var id_title = "<h4>" + ids[0].toString().replace(/<br>/g, ' ') + ': ' + titles[0].toString().replace(/<br>/g, ' ') + "</h4>";
            var publication = "<p1><b>Published: </b> " + published_times[0].toString().replace(/<br>/g, ' ') + "<br>";
            // var full_text = "<h4>" + texts[0].toString().replace(/<br>/g, ' ') + "</h4>";
            // var link = "<b>Link:</b> <a href='" + links[0].toString() + "' target='_blank'>" + links[0].toString() + "</a></p1>"

            if (links[0].toString() == 'No link available') {
                var link = "<b>Link: </b> " + links[0].toString() + "</p1>"
            } else {
                var link = "<b>Link: </b> <a href='" + links[0].toString() + "' target='_blank'>" + links[0].toString() + "</a></p1>"
            }

            current_selection.text = id_title + publication + link
            current_selection.change.emit();
    """
    return code

# handle the keywords and search
def input_callback(plot, source, out_text, topics, nr_of_topics): 

    # slider call back for cluster selection
    callback = CustomJS(args=dict(p=plot, source=source, out_text=out_text, topics=topics, nr_of_topics=nr_of_topics), code="""
				var key = text.value;
				key = key.toLowerCase();
				var cluster = slider.value;
                var data = source.data; 
                
                var x = data['x'];
                var y = data['y'];
                var x_backup = data['x_backup'];
                var y_backup = data['y_backup'];
                var labels = data['desc'];
                var text = data['text'];
                var titles = data['titles'];
                if (cluster == nr_of_topics.toString()) {
                    out_text.text = 'Keywords: Slide to specific cluster to see the keywords.';
                    for (var i = 0; i < x.length; i++) {
						if(text[i].includes(key) || 
						titles[i].includes(key)) {
							x[i] = x_backup[i];
							y[i] = y_backup[i];
						} else {
							x[i] = undefined;
							y[i] = undefined;
						}
                    }
                }
                else {
                    out_text.text = 'Keywords: ' + topics[Number(cluster)];
                    for (var i = 0; i < x.length; i++) {
                        if(labels[i] == cluster) {
							if(text[i].includes(key) || 
							titles[i].includes(key)) {
								x[i] = x_backup[i];
								y[i] = y_backup[i];
							} else {
								x[i] = undefined;
								y[i] = undefined;
							}
                        } else {
                            x[i] = undefined;
                            y[i] = undefined;
                        }
                    }
                }
            source.change.emit();
            """)
    return callback

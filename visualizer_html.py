import os
import plotly.graph_objects as go

def visualize_and_save_frame_with_belief(frame, title, epoch, image_type, belief, probability,
                                         output_dir):
    x = frame[:, 0]
    y = frame[:, 2]
    z = -frame[:, 1]  # Correcting the orientation here

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color='red')
    )

    pairs = [(0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10), (4, 5), (1, 2), (5, 6),
             (2, 3), (8, 11), (8, 14), (11, 12), (14, 15), (12, 13), (15, 16)]

    lines = [go.Scatter3d(
        x=[x[pair[0]], x[pair[1]]],
        y=[y[pair[0]], y[pair[1]]],
        z=[z[pair[0]], z[pair[1]]],
        mode='lines',
        line=dict(color='green', width=2)
    ) for pair in pairs]


    # Adding text annotation for discriminator's belief, probability, and additional details
    annotations = [
        dict(
            showarrow=False,
            x=sum(x) / len(x), y=sum(y) / len(y), z=sum(z) / len(z),
            text=f"{title}, Epoch: {epoch}, Type: {image_type}, Belief: {belief}, Probability: {probability:.2f}",
            xanchor="left",
            xshift=10,
            font=dict(color="black", size=14)
        ),
        dict(
            showarrow=False,
            x=sum(x) / len(x), y=min(y), z=min(z),  # Positioning this at the bottom
            text=f"{title}, Epoch: {epoch}, Type: {image_type}, Belief: {belief}, Probability: {probability:.2f}",
            xanchor="left",
            yanchor="bottom",
            font=dict(color="blue", size=12)
        )
    ]

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', nticks=10, range=[min(x), max(x)]),
            yaxis=dict(title='Y', nticks=10, range=[min(y), max(y)]),
            zaxis=dict(title='Z', nticks=10, range=[min(z), max(z)]),
            annotations=annotations
        ),
        margin=dict(l=0, r=0, b=0, t=30)  # Adjust margins to make room for title and details
    )

    fig = go.Figure(data=[scatter] + lines, layout=layout)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"{epoch}_{image_type}_{belief}_{probability:.2f}.html")
    fig.write_html(filename)

    return filename
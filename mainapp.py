from turtle import heading, width
from urllib import response
import dash
from dash import html
from dash.dependencies import Input, Output, State
from flask import Flask, Response, stream_with_context
import cv2
import mediapipe as mp
import dash_core_components as dcc
import numpy as np
from PIL import Image
import io
import base64
import plotly.express as px
import tensorflow as tf
import time


model = tf.keras.models.load_model('model_1408')
app = dash.Dash(__name__)


app.layout = html.Div([
    
    html.Div([
        html.Div(style = {'width':'10%', 'height':'100%'}),
        html.Div([
            html.H1("Fruit center detection"),
            html.Br(),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Div("Please upload the movement video here"),
                    html.Div("Drag or upload from local device"),
                ]),
                style={
                    'width': '100%',
                    'height': '120px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            html.Div(id = "download-check"),
            html.Br(),
            html.Br(),  
            html.Button('Start', id='video-start-button', n_clicks=0, style = {"width":"100px", "height":"50px", "font-size":"20px"})
        ], style = {'width':'40%', 'height':'100%'}),
        html.Div([""], style = {'width':'5%'}),
        html.Div([
            html.Br(),
            html.Br(), 
            dcc.Loading(id = 'pic-loading',
                        children = html.Div(["testing"], id = 'video-output'),
                        type = 'circle'
            ),
        ], style = {'width':'40%', 'height':'100%'}),
        html.H1(style = {'width':'10%', 'height':'100%'}),
    ], style = {'display':'flex'})
])

def center_prediction_pipeline(im, model):
    im=im
    origin_w = im.shape[1]
    origin_h = im.shape[0]
    resized_image = cv2.resize(im, (244, 244))
    normalized_image = resized_image/255
    Image = np.expand_dims(normalized_image, axis=0)
    prediction = model.predict(Image)
    RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = cv2.circle(RGB_im, (int(prediction[0][0]*origin_w),int(prediction[0][1]*origin_h)), radius=25, color=(255, 0, 0), thickness = -1)
    fig = px.imshow(image)
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return fig
 
@app.callback(
    Output('download-check', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
    )

def video_display(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = html.Div("Successfully load!")
        return children
    else:
        children = html.Div("Load something")
        return children

@app.callback(
    Output('video-output', 'children'),
    Input('video-start-button', 'n_clicks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
    )


def video_display(n_clciks, list_of_contents, list_of_names, list_of_dates):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if ('video-start-button' in  changed_id) and list_of_contents is not None:
        time.sleep(5)
        if list_of_names[-3:] == 'png':
            data = list_of_contents.replace('data:image/png;base64,', '')
        elif list_of_names[-3:] == 'jpg':
            data = list_of_contents.replace('data:image/jpeg;base64,', '')
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        im = np.array(img)
        RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        fig = center_prediction_pipeline(RGB_im, model)
        return_div = dcc.Graph(figure = fig, style = {'width':'800px', 'height':'800px'}) 
    else:
        return_div = html.Div("Please upload and start the analysis")

    return [return_div]

if __name__ == '__main__':
    app.run_server(debug=False)
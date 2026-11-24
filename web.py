from flask import Flask, render_template_string, request
from PIL import Image, UnidentifiedImageError
from test_pkl_model import classify_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>INDIMON</title>
<style>body{max-width:38rem;margin:4rem auto;padding:0 1rem;font:1.1rem/1.5 system-ui;background:#f6efe4;color:#24231f}label,input,button{display:block;margin:1rem 0}button{font:inherit;padding:.55rem 1rem}</style>
<main><h1>INDIMON</h1><p>Try the bundled Indian monument classifier.</p>
<form method="post" enctype="multipart/form-data"><label for="photo">Choose a JPEG or PNG</label><input id="photo" name="photo" type="file" accept="image/jpeg,image/png" required><button>Classify photo</button></form>
{% if result %}<p role="status">{{ result }}</p>{% endif %}</main></html>'''


@app.errorhandler(413)
def too_large(_error):
    return render_template_string(PAGE, result='Choose an image smaller than 8 MB.'), 413


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(PAGE)
    photo = request.files.get('photo')
    if not photo or not photo.filename:
        return render_template_string(PAGE, result='Choose a JPEG or PNG image.'), 400
    try:
        with Image.open(photo.stream) as image:
            if image.format not in {'JPEG', 'PNG'}:
                return render_template_string(PAGE, result='Choose a JPEG or PNG image.'), 400
        photo.stream.seek(0)
        label, score = classify_image(photo.stream)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return render_template_string(PAGE, result='The image could not be read.'), 400
    if score < 0.50:
        result = 'Unknown image'
    elif label == 'not_monument':
        result = 'No monument identified'
    else:
        result = ('Possible match: ' if score < 0.75 else '') + label.replace('_', ' ')
    return render_template_string(PAGE, result=f'{result} ({score:.1%} model confidence)')


if __name__ == '__main__':
    app.run()

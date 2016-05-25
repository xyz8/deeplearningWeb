from flask import render_template
from flask_wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from . import main


class NameForm(Form):
    name = StringField('put your sentence?', validators=[Required()])
    submit = SubmitField('Submit')

@main.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        form.name.data = ''
    return render_template('index.html', form=form, name=name)

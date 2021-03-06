#coding:utf-8
from flask import render_template,Flask
from flask_wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from test_model import *
from sentiment_model import train,test
from flask_bootstrap import Bootstrap
from flask_moment import Moment
app = Flask('__name__')
app.config['SECRET_KEY']='xxx'
bootstrap=Bootstrap(app)
moment=Moment(app)

#单独一个句子的表单
class SingleForm(Form):
    sen = StringField('',validators=[Required()])
    submit = SubmitField('提交')

#输入3个文件路径的表单
class FileForm(Form):
    before = StringField('原始文件路径',validators=[Required()])
    fengci = StringField('分词文件路径',validators=[Required()])
    sens = StringField('标签文件路径',validators=[Required()])
    submit = SubmitField('提交')

#输入相似词的表单
class wordForm(Form):
    word = StringField('',validators=[Required()])
    submit = SubmitField('提交')

#输入训练和测试参数的表单
class trainForm(Form):
    hidden_node1 = StringField('第1层隐含层节点数',validators=[Required()])
    hidden_node2 = StringField('第2层隐含层节点数',validators=[Required()])
    hidden_node3 = StringField('第3层隐含层节点数',validators=[Required()])
    batch = StringField('每次迭代的样本数目',validators=[Required()])
    nb = StringField('迭代次数',validators=[Required()])
    loss_way =  StringField('优化方法',validators=[Required()])
    loss_object = StringField('目标函数',validators=[Required()])
    train_path =  StringField('训练样本',validators=[Required()])
    submit = SubmitField('提交')

class testForm(Form):
    test_path =  StringField('测试样本',validators=[Required()])
    submit = SubmitField('提交')

@app.route('/', methods=['GET', 'POST'])
def index():
    sen = None
    Singleform = SingleForm()
    result = None
    Fileform = FileForm()
    if Singleform.validate_on_submit():
        sen = Singleform.sen.data
	sen = predic_sens(sen)
        sen = 'negative' if sen == '[[0]]' else 'positive'
	return render_template('index.html',Singleform=Singleform,sen=sen,Fileform=Fileform,result=result)
    if Fileform.validate_on_submit():
        doc = Fileform.before.data
        fengci = Fileform.fengci.data
        sens = Fileform.sens.data
	result = predict_file(doc,fengci,sens)
    	return render_template('index.html',Singleform=Singleform,sen=sen,Fileform=Fileform,result=result)
    return render_template('index.html',Singleform=Singleform,sen=sen,Fileform=Fileform,result=result)

@app.route('/similarWord', methods=['GET', 'POST'])
def similarWord():
    words = None
    word_list = []
    word_form = wordForm()
    if word_form.validate_on_submit():
        word = word_form.word.data
        words = list(find_similar(word))
    return render_template('similarWord.html',word_form=word_form,words=words)
	
@app.route('/trainTest', methods=['GET', 'POST'])
def trainTest():
    hidden_node1,hidden_node2,hidden_node3, batch,nb,loss_way,loss_object = None,None,None,None,None,None,None
    train_path, test_path = None, None
    isTrain, isTest = False, False
    train_form = trainForm()
    test_form = testForm()
    if train_form.validate_on_submit():
        hidden_node1,hidden_node2,hidden_node3 = int(train_form.hidden_node1.data),int(train_form.hidden_node2.data),int(train_form.hidden_node3.data)
        batch,nb,loss_way,loss_object = int(train_form.batch.data),int(train_form.nb.data),train_form.loss_way.data,train_form.loss_object.data      
        train_path = train_form.train_path.data
        train(train_path,hidden_node1,hidden_node2,hidden_node3, batch,nb,loss_way,loss_object)
        isTrain = True
    elif test_form.validate_on_submit():
        test_path = test_form.test_path.data
        test(test_path)
        isTest = True
    return render_template('trainTest.html',train_form=train_form,test_form=test_form,isTrain=isTrain,isTest=isTest)

if __name__ == "__main__":
	app.run(debug=True)

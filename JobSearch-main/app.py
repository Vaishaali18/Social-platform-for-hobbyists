from flask import Flask,request,jsonify,render_template,redirect,url_for,Response
from flask_mysqldb import MySQL
import shortuuid
import base64
import sys
from igraph import Graph
import numpy as np
import os
import glob
import random
from random import shuffle
from random import seed
import matplotlib.pyplot as plt
import time
import datetime
import collections

app = Flask(__name__)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '18vaishaali'
app.config['MYSQL_DB'] = 'hobify'

mysql = MySQL(app)

# cur.execute("select image from posts where stud_id=%s",(id,))
 #          res=cur.fetchall()
  #         if res:
   #            res=res+","+encoded_string 
    #           cur.execute("update posts set image=%s where stud_id=%s",(id,string1))
     #          mysql.connection.commit()
      #         cur.close()

def get_edge_list(dataset_path):
    data_file = open(dataset_path)
    edge_list = map(lambda x:tuple(map(int,x.split())),data_file.read().split("\n")[:-1])
    data_file.close()
    return edge_list

# Get the similarity product for a path
# (product of path-step similarities)
def get_sim_product(sim, shortest_path):
    prod = 1
    for i in range(len(shortest_path) - 1):
        prod *= sim[shortest_path[i]][shortest_path[i+1]]
    return round(prod,3)

# Filter out, Sort and Get top-K predictions
# Filter out, Sort and Get top-K predictions
def get_top_k_recommendations(graph,sim,i,k):
    return  sorted(filter(lambda x: i!=x and graph[i,x] != 1,range(len(sim[i]))) , key=lambda x: sim[i][x],reverse=True)[0:k]

# Convert edge_list into a set of constituent edges
def get_vertices_set(edge_list):
    res = set()
    for x,y in edge_list:
        res.add(x)
        res.add(y)
    return res

# Split the dataset into two parts (50-50 split)
# Create 2 graphs, 1 used for training and the other for testing
def split_data(edge_list):
    random.seed(350)
    li=list(edge_list)
    indexes = range(len(li))
    test_indexes = set(random.sample(indexes, int(len(indexes)/2))) # removing 50% edges from test data
    train_indexes = set(indexes).difference(test_indexes)
    test_list = [li[i] for i in test_indexes]
    train_list = [li[i] for i in train_indexes]
    return train_list,test_list

# Calculates accuracy metrics (Precision & Recall),
# for a given similarity-model against a test-graph.

def print_precision_and_recall(sim,graph,test_graph,test_vertices_set,train_vertices_set):
    i=id
    k=12
    top_k=[]
    try:
        top_k = list(sorted(filter(lambda x: i!=x and graph[i,x] != 1,range(len(sim[i]))) , key=lambda x: sim[i][x],reverse=True)[0:k])
        top_k.sort()
        return top_k
    except IndexError:
        return top_k
    except TypeError:
        return top_k

'''L=["20"," ","16","\n"]
file1 = open("C:/Users/vaishaali/Downloads/Link_prediction_social_network-master/Link_prediction_social_network-master/data/f.txt","a")
file1.writelines(L)'''
            


# http://be.amazd.com/link-prediction/
def similarity(graph, i, j, method):
    if method == "common_neighbors":
        return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))
    elif method == "jaccard":
        return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))/float(len(set(graph.neighbors(i)).union(set(graph.neighbors(j)))))
    elif method == "adamic_adar":
        return sum([1.0/math.log(graph.degree(v)) for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])
    elif method == "preferential_attachment":
        return graph.degree(i) * graph.degree(j)



###################################
### Methods for Link Prediction ###
###################################
def local_methods(edge_list,method):
    train_list, test_list = split_data(edge_list)
    train_graph = Graph(train_list)
    test_graph = Graph(test_list)
    train_n =  train_graph.vcount() # This is maximum of the vertex id + 1
    train_vertices_set = get_vertices_set(train_list) # Need this because we have to only consider target users who are present in this train_vertices_set
    test_vertices_set = get_vertices_set(test_list) # Set of target users

    sim = [[0 for i in range(train_n)] for j in range(train_n)]
    for i in range(train_n):
        for j in range(train_n):
            if i!=j and i in train_vertices_set and j in train_vertices_set:
                sim[i][j] = similarity(train_graph,i,j,method)

    top_k=print_precision_and_recall(sim,train_graph,test_graph,test_vertices_set,train_vertices_set)
    return top_k

def rec():
    # default-case/ help
    # Command line argument parsing
    dataset_path='C:/Users/vaishaali/Documents/18C118/JobSearch-main/f.txt'
    method = "common_neighbors"
    
    edge_list = get_edge_list(dataset_path)

    if method == "common_neighbors" or method == "jaccard" or method == "adamic_adar" or method == "preferential_attachment":
        top_k=local_methods(edge_list,method)
        return top_k

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        global id
        user=request.form['user']
        password=request.form['pass']
        cur = mysql.connection.cursor()
        cur.execute("select pass,stud_id FROM details1 WHERE email=%s",(user,))
        res=cur.fetchall()
        if res[0][0]==password:
            id=res[0][1]
            mysql.connection.commit()
            cur.close()
            return redirect(url_for("next"))
        else:
            return render_template('login.html',err="Invalid Credentials")
       

@app.route('/posts', methods=['GET', 'POST'])
def posts():
       if request.method == "POST":
            global id
            details = request.form
            firstName = details['fname']
            lastName = details['lname']
            phone=details['phone']
            loc=details['loc']
            hobbies=request.form.getlist("hob")
            str1 = ','.join(hobbies)
            email=details["email"]
            password=details["pass"]
            cur = mysql.connection.cursor()
            cur1 = mysql.connection.cursor()
            cur.execute("INSERT INTO details1(firstname,lastname,phone,location,hobbies,email,pass) VALUES (%s, %s,%s,%s,%s,%s,%s)", (firstName, lastName,phone,loc,str1,email,password))
            cur1.execute("select stud_id from details1 where email=%s",(email,))
            res1=cur1.fetchone()
            id=res1[0]
            cur.execute("insert into dp(stud_id,pic) values(%s,%s)",(id,"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"))
            mysql.connection.commit()
            cur.close()
            cur1.close()
            return redirect(url_for("next"))


@app.route('/reg')
def reg():
    return render_template("register.html")

@app.route('/next')
def next():
    cur = mysql.connection.cursor()
    cur.execute("select p2 from followers where p1=%s",(id,))
    res=cur.fetchall()
    global top_k1
    top_k=rec()
    if len(top_k)==0:
        top_k=[]
        if len(res)==0:          
            for i in range(1,total):
                top_k.append(i)
        else:
            li=[]
            for i in res:
                li.append(i[0])
            for i in range(1,total):
                if i not in li:
                    top_k.append(i)
            
        top_k1=tuple(top_k)
    else:
        if 0 in top_k:
            top_k.remove(0)
        if len(res)!=0:
            for i in res:
                if i[0] in top_k:
                    top_k.remove(i[0])
        
        top_k1=tuple(top_k)
    print(top_k1)
    cur.execute("select p.comments,p.image,d.firstname,d.lastname from posts p inner join details1 d on p.stud_id=d.stud_id where p.stud_id<>%s and p.stud_id in %s",(id,top_k1,))
    res1=cur.fetchall()
    cur.execute("select p.comments,p.image,d.firstname,d.lastname from posts p inner join details1 d on p.stud_id=d.stud_id where p.stud_id<>%s and p.stud_id not in %s",(id,top_k1,))
    res2=cur.fetchall()
    mysql.connection.commit()
    cur.close()
    return render_template("home1.html",show=1,res2=res2,res1=res1,len1=len(res1),len2=len(res2))

@app.route('/filter1',methods=['GET', 'POST'])
def filter1():
    if request.method == "POST":
        shob=request.form.getlist('shob')
        cur = mysql.connection.cursor()
        cur.execute("select p.comments,p.image,d.firstname,d.lastname from posts p inner join details1 d on p.stud_id=d.stud_id where p.category in %s",(shob,))
        res1=cur.fetchall()
        mysql.connection.commit()
        cur.close()
        return render_template("home1.html",res1=res1,len1=len(res1),show=0)

@app.route('/addnext')
def addnext():
    return render_template("adding.html")

@app.route('/edit')
def edit():
    return render_template("edit.html")

@app.route('/edit1',methods=['GET', 'POST'])
def edit1():
     if request.method == "POST":   
           image=request.files['uploadimage']
           encoded_string= base64.b64encode(image.read())
           cur = mysql.connection.cursor()              
           cur.execute("UPDATE dp SET pic=%s WHERE stud_id=%s", (encoded_string,id))
           mysql.connection.commit()
           cur.close()
           return redirect(url_for("prof"))

@app.route('/foll',methods=['GET', 'POST'] )
def foll():
  if request.method == "GET":
    cur = mysql.connection.cursor()   
    cur.execute("select count(*) from followers where p1=%s",(id,))
    res=cur.fetchall()
    followers=res[0][0]
    print(followers)
    cur.execute("select count(*) from details1")
    res=cur.fetchall();
    total=res[0][0]
    cur.execute("select p2 from followers where p1=%s",(id,))
    res=cur.fetchall()
    global top_k1
    top_k=rec()
    if len(top_k)==0:
        top_k=[]
        if len(res)==0:          
            for i in range(1,total):
                top_k.append(i)
        else:
            li=[]
            for i in res:
                li.append(i[0])
            for i in range(1,total):
                if i not in li:
                    top_k.append(i)
            
        top_k1=tuple(top_k)
    else:
        if 0 in top_k:
            top_k.remove(0)
        if len(res)!=0:
            for i in res:
                if i[0] in top_k:
                    top_k.remove(i[0])
        
        top_k1=tuple(top_k)
        
    cur = mysql.connection.cursor()
    cur1=mysql.connection.cursor()
    cur.execute("select d.firstname,d.lastname,d.hobbies,p.pic,d.stud_id from details1 d inner join dp p on d.stud_id=p.stud_id where d.stud_id in %s",(top_k1,))    
    res1=cur.fetchall()
    cur1.execute("select d.firstname,d.lastname,d.hobbies,p.pic,d.stud_id from details1 d inner join dp p on d.stud_id=p.stud_id where d.stud_id not in %s and d.stud_id <> %s",(top_k1,id,))
    res2=cur1.fetchall()
    res2=list(res2)
    follist=[]
    for i in res:
        follist.append(i[0])
    follist.sort()
    for i in res2:
        print(i[4])
    for i in res2:
        if i[4] in follist:
            res2.remove(i)
    print(follist)
    for i in res2:
        print(i[4])
    res2=tuple(res2)
    if len(res2)>0:
        display=1
    else:
        display=0
    return render_template("follow.html",display=display,res1=res1,len=len(res1),res2=res2,len2=len(res2))    

@app.route('/profileview',methods=['GET', 'POST'])
def profileview():
    if request.method == "POST":  
        id1=request.form['id']
        print(id1)
        cur = mysql.connection.cursor()
        cur.execute("select count(*) from posts where stud_id=%s",(id1,))
        res=cur.fetchall()
        postcount=res[0][0]
        cur.execute("select count(*) from followers where p1=%s",(id1,))
        res=cur.fetchall()
        followers=res[0][0]
        cur.execute("select count(*) from followers where p2=%s",(id1,))
        res=cur.fetchall()
        following=res[0][0]
        cur.execute("select d.firstname,d.lastname,d.hobbies,p.pic,d.stud_id from details1 d inner join dp p on d.stud_id=p.stud_id where d.stud_id=%s",(id1,))
        res1=cur.fetchall()
        name=res1[0][0]+" "+res1[0][1]
        hobbies=res1[0][2]
        hob=[]
        elements =list(hobbies.split(","))
        for i in range(0,len(elements)):
            if elements[i]=="1":
                hob.append("Drawing")
            elif elements[i]=="2":
                hob.append("Painting")
            elif elements[i]=="3":
                hob.append("Gardening")
            elif elements[i]=="4":
                hob.append("Sports")
        return render_template('profview.html',res1=res1,name=name,hob=hob,len=len(hob),pc=postcount,f1=followers,f2=following)     

@app.route('/addfollow',methods=['GET', 'POST'])
def addfollow():
    if request.method == "POST": 
        followid=request.form['followid']
        L=[str(id)," ",followid,"\n"]
        file1 = open("f.txt","a")
        file1.writelines(L)
        cur = mysql.connection.cursor()
        print(str(id)+" "+followid)
        cur.execute("insert into followers(p1,p2) values(%s,%s)",(id,int(followid)))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for("foll"))

@app.route('/prof')
def prof():
    cur = mysql.connection.cursor()
    cur.execute("select count(*) from followers where p1=%s",(id,))
    res=cur.fetchall()
    followers=res[0][0]
    cur.execute("select count(*) from followers where p2=%s",(id,))
    res=cur.fetchall()
    following=res[0][0]
    cur1 = mysql.connection.cursor()
    cur2 = mysql.connection.cursor()
    cur.execute("select firstname,lastname,location,hobbies,email from details1 where stud_id=%s",(id,))
    cur1.execute("select image from posts where stud_id=%s",(id,))
    cur2.execute("select pic from dp where stud_id=%s",(id,))
    res1=cur.fetchall()
    res2=cur1.fetchall()
    res3=cur2.fetchone()
    name=res1[0][0]+" "+res1[0][1]
    loc=res1[0][2]
    hobbies=res1[0][3]
    email=res1[0][4]
    hob=[]
    elements =list(hobbies.split(","))
    for i in range(0,len(elements)):
        if elements[i]=="1":
            hob.append("Drawing")
        elif elements[i]=="2":
            hob.append("Painting")
        elif elements[i]=="3":
            hob.append("Gardening")
        elif elements[i]=="4":
            hob.append("Sports")
    return render_template("profile.html",followers=followers,following=following,res3=res3,len1=len(res2),res2=res2,len=len(hob),name=name,loc=loc,hob=hob,email=email)
         
@app.route('/addprof',methods=['GET', 'POST'])
def addprof():
     if request.method == "POST":
           comments=request.form['comments'] 
           categ=request.form['categ']    
           image=request.files['addimage']
           encoded_string= base64.b64encode(image.read())
           cur = mysql.connection.cursor()              
           cur.execute("INSERT INTO posts(stud_id,category,comments,image) VALUES (%s,%s,%s,%s)", (id,categ,comments,encoded_string))
           mysql.connection.commit()
           cur.close()
           return redirect(url_for("prof"))
     else:
           return redirect(url_for("prof"))

@app.route('/forum',methods=['GET', 'POST'])
def forum():
    return render_template('forum.html')

@app.route('/topic',methods=['GET', 'POST'])
def topic():
    global threadid
    threadid = request.args['id']
    cur = mysql.connection.cursor()
    cur.execute("select tname,description from thread where thread_id=%s",(threadid,))
    res=cur.fetchall()
    cur.execute("select img from background where num=%s",(threadid,))
    res1=cur.fetchall()
    return render_template('thread.html',threadid=threadid,res=res,len=len(res),res1=res1)

@app.route('/newtopic',methods=['GET', 'POST'])
def newtopic():
     if request.method == "POST":
         topic=request.form['topic']
         desc=request.form['desc']
         cur = mysql.connection.cursor()
         cur.execute("insert into thread values(%s,%s,%s)",(threadid,topic,desc))
         mysql.connection.commit()
         cur.close()
         return redirect(url_for("topic",id=threadid))

@app.route('/discuss',methods=['GET', 'POST'])
def discuss():
    global discussid
    discussid = request.args['id']
    cur = mysql.connection.cursor()
    cur.execute("select comments from discuss where topic=%s and thread=%s",(threadid,discussid,))
    res=cur.fetchall()
    cur.execute("select p.pic from dp p inner join discuss d on d.stud_id=p.stud_id where topic=%s and thread=%s",(threadid,discussid,))
    res1=cur.fetchall()
    cur.execute("select pic from dp where stud_id=%s",(id,))
    res3=cur.fetchall()
    cur.execute("select img from background where num=%s",(threadid,))
    res2=cur.fetchall()
    return render_template('discuss.html',res=res,len=len(res),res1=res1,res2=res2,res3=res3)

@app.route('/reply',methods=['GET', 'POST'])
def reply():
     if request.method == "POST":
         reply=request.form['reply']
         cur = mysql.connection.cursor()
         cur.execute("insert into discuss(topic,thread,comments,stud_id) values(%s,%s,%s,%s)",(threadid,discussid,reply,id))
         mysql.connection.commit()
         cur.close()
         return redirect(url_for("discuss",id=discussid))

if __name__ == '__main__':  
    app.run(debug = True) 

    





  

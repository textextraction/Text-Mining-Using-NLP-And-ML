import requests
from Bio import Entrez
import urllib
import json
import pymysql
import dateutil.parser
import datetime
import bs4 as bs
from bs4 import BeautifulSoup
import random
import re
import operator
import nltk
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

Count = 0
pmid = 0
flag = 0
flaga = 0
doi = " "
pmcid = " "
author = " "
lstr = []
topl=[]
counts={}

def TermCall():
    print(" ")
    Term = raw_input("Enter the Term you require: ")
    print(" ")
    return Term
    
def SearchCall(number,flag,Term):
    r = 0
    Entrez.email = "name@example.com"
    if(flag==1):
        r = random.randint(0,number)
    handle = Entrez.esearch(db="pmc", term=Term, retstart = r, retmax=number)
    record = Entrez.read(handle)
    count = record['Count']
    IDs = (record['IdList'])
    if(flag==0):
        return count
        flag =1
    else:
        return IDs
    handle.close()

    
def CountConsider(count):
    if(count==0):
        print("No Papers Regarding the search item!")
        return
    else:
        print(" ")
        print(" ")
        print ("The Term contains These many number of papers: "+count)

        print(" ")
        print(" ")
        Count = input("Enter the number of papers you want to consider: ")
        return Count
    
def MetadataCall(ids):
    response = urllib.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id="+ids+"&retmode=json")
    data = json.loads(response.read())

    print("PMCID: "+ids)
    print(" ")

    print("JSON LINK: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id="+ids+"&retmode=json")
    print(" ")

    print("PAPER LINK: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"+ids+"/")
    print(" ")
        
    return data

def MetadataParsing(data):
    metadatalist = list()
    if('header' in data):
        if('version' in data['header']):
            version = data['header']['version']
        else:
            version = " "
        metadatalist.append(version)
        print("Version: "+version)

        if('type' in data['header']):
            type = data['header']['type']
        else:
            type = " "
        metadatalist.append(type)
        print("Type: "+type)

        if('result' in data):
            if('uids' in data['result']):
                uids = data['result']['uids']
                uids = uids[0]
            else:
                uids = " "
            print("uids: "+uids)
            metadatalist.append(uids)
            
            if(uids in data['result']):
                if('sortdate' in data['result'][uids]):
                    sortdate = data['result'][uids]['sortdate']
                    sortDate = dateutil.parser.parse(sortdate).date()
                    sortDate = str(sortDate)
                    #sortDate = datetime.datetime.strptime(sortDate, '%Y-%m-%d')
                    #sortDate = datetime.date.strftime(sortDate, "%Y/%m/%d")
                else:
                    sortdate = " "
                    sortDate = " "
                metadatalist.append(sortDate)
                print("SortDate: "+sortdate)

                if('fulljournalname' in data['result'][uids]):
                    fulljournalname = data['result'][uids]['fulljournalname']
                else:
                    fulljournalname = " "
                metadatalist.append(fulljournalname)
                print("FulljournalName: "+fulljournalname)

                if('pubdate' in data['result'][uids]):
                    pubdate = data['result'][uids]['pubdate']
                    pubDate = str(pubdate)
                else:
                    pubdate = " "
                    pubDate = " "
                metadatalist.append(pubDate)
                print("Pubdate: "+pubdate)

                if('title' in data['result'][uids]):
                    title = data['result'][uids]['title']
                else:
                    title = " "
                metadatalist.append(title)
                print("Title: "+title)

                if('printpubdate' in data['result'][uids]):
                    printpubdate = data['result'][uids]['printpubdate']
                else:
                    printpubdate = " "
                metadatalist.append(printpubdate)
                print("Printpubdate: "+printpubdate)

                if('articleids' in data['result'][uids]):
                    articleids = data['result'][uids]['articleids']

                    lent = len(articleids)
                    for i in range(0,lent):
                        articleids1 = articleids[i]
                        if('idtype' in articleids1):
                            idtype = articleids1['idtype']
                        else:
                            idtype = " "

                        if('value' in articleids1):
                            value = articleids1['value']
                        else:
                            value = " "
                        if(idtype=='pmid'):
                            pmid = value
                            metadatalist.append(pmid)
                        if(idtype=='doi'):
                            doi = value
                            metadatalist.append(doi)
                        if(idtype=='pmcid'):
                            pmcid = value
                            metadatalist.append(pmcid)
                        print(idtype+" :"+value)

                if('pmclivedate' in data['result'][uids]):
                    pmclivedate = data['result'][uids]['pmclivedate']
                    pmcliveDate = str(pmclivedate)
                else:
                    pmclivedate = " "
                    pmcliveDate = " "
                metadatalist.append(pmcliveDate)
                print("Pmclivedate: "+pmclivedate)

                if('volume' in data['result'][uids]):
                    volume = data['result'][uids]['volume']
                else:
                    volume = " "
                metadatalist.append(volume)    
                print("Volume: "+volume)

                if('source' in data['result'][uids]):
                    source = data['result'][uids]['source']
                else:
                    source = " "
                metadatalist.append(source)
                print("Source: "+source)

                if('epubdate' in data['result'][uids]):
                    epubdate = data['result'][uids]['epubdate']
                else:
                    epubdate = " "
                metadatalist.append(epubdate)
                print("Epubdate: "+epubdate)

                if('authors' in data['result'][uids]):
                    authors = data['result'][uids]['authors']
                    lent = len(authors)
                    for i in range(0,lent):
                        authors1 = authors[i]
                        if('authtype' in authors1):
                            authtype = authors1['authtype']
                        else:
                            authtype = " "
                        if(authtype == 'Author'):
                            if('name' in authors1):
                                name = authors1['name']
                            else:
                                name = " "

                            if(i==0):
                                author = name+","
                            if(i==(lent-1)):
                                author = author+name
                            else:
                                author = author+name+","
                    metadatalist.append(author)
                    print("Authors: "+author)

                if('issue' in data['result'][uids]):
                    issue = data['result'][uids]['issue']
                    issue = str(issue)
                else:
                    issue = " "
                metadatalist.append(issue)
                print("Issue: "+issue)

                if('pages' in data['result'][uids]):
                    pages = data['result'][uids]['pages']
                    pages = str(pages)
                else:
                    pages = " "
                metadatalist.append(pages)
                print("Pages: "+pages)

        print('')
        print(" ")
    return metadatalist

def DatabaseConnect(mlist,q,cursor,db):

    try:
        cursor.execute("insert into metadata(ID,Version,Type,UID,Sortdate,Fulljournalname,Pubdate,Title,Printpubdate,PMID,DOI,PMCID,Pmclivedate,Volume,Source,Epubdate,Author,Issue,Pages)values("+q+","+mlist[0]+",'"+mlist[1]+"',"+mlist[2]+",'"+mlist[3]+"','"+mlist[4]+"','"+mlist[5]+"','"+mlist[6]+"','"+mlist[7]+"',"+mlist[8]+",'"+mlist[9]+"','"+mlist[10]+"','"+mlist[11]+"',"+mlist[12]+",'"+mlist[13]+"','"+mlist[14]+"','"+mlist[15]+"','"+mlist[16]+"','"+mlist[17]+"')")        
        db.commit()
    except:
        db.rollback()
        print(" ")
        print(" ")
        print(" ")
        print("Rollbacked in main!")
    

def AbstractParsing(flaga,ids):
    sauce=urllib.urlopen('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id='+ids)
    soup = bs.BeautifulSoup(sauce,'lxml')
    for paragraph in soup.find_all('abstract'):
        abstract = paragraph.text
        if(flaga!=1):
            print(abstract)
            print("")
            print('')
    if(flaga==1):
        return abstract
    
def ReferencesParsing(ids,mlist,cursor,db):
    page = requests.get("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"+ids+"/")
    soup = BeautifulSoup(page.content, 'html.parser')
    if(soup==""):
        print("No references extracted!")
    else:
        m = 0
        for paragraph in soup.find_all('span', class_ ='element-citation'):
            m = m+1
            n = str(m)
            pstr = "                                                 Reference number "+n+" : "
            print(pstr)
            print("")
            print("")
            print("")

            w=paragraph.text
            parts = w.split(".")

            w=parts[0]
            print("Authors:")
            print(w) 

            print("")
            print("Paper title:")
            print(parts[1])
            print("")

            print("Journal:")
            print(parts[2])

            print("year")
            print(parts[3])    

            print("")

            
            (parts[2]) = (parts[2]).encode('ascii','ignore')
            (parts[1]) = (parts[1]).encode('ascii','ignore')
            (parts[0]) = (parts[0]).encode('ascii','ignore')
            (parts[3]) = (parts[3]).encode('ascii','ignore')

            try:
                cursor.execute("INSERT INTO referencestable(PMCID,PMID,Author,Title,ReferencePaperTitle,ReferenceAuthors)VALUES('"+mlist[10]+"','"+mlist[8]+"','"+mlist[15]+"','"+mlist[6]+"','"+parts[1]+"','"+parts[0]+"')")
                db.commit()
            except:
                db.rollback()
                print(" ")
                print("Rollbacked in reference!")
                print(" ")
    
def FetchAuthor(db,cursor):
    cursor.execute("SELECT * FROM referencestable")
    records = cursor.fetchall()
    row_count = cursor.rowcount
    for row in records:
        string2 = row[5]
        string2 = string2.replace(", et al","")
        string2 = string2.replace(", etal","")
        #string1 = re.sub(" ","",row[2])
        #string2 = re.sub(" ","",string2)
        #string0 = string1+","+string2
        string = string2.split(",")
        strc = len(string)
        for i in string:
            if(strc!=0):
                lstr.append(i)
                strc = strc - 1
        #print "PMCID: "+str(row[0])
        #print "Author: "+str(row[2])
        #print "Reference Author: "+str(row[5])
    return lstr

def AuthorCount(lstr):
    keyv=[]
    valv=[]
    n = 0
    for word in lstr:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    countl = len(counts)
    for key, value in sorted(counts.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)
        n=n+1
        if n>(countl-20) and n<(countl-1):
            keyv.append(key)
            valv.append(value)
    plt.pie(valv, labels=keyv, startangle=90, autopct='%.1f%%')
    plt.title('Importance of a Research Articles: Basis: Referred Authors')
    plt.show()
    return counts

def TopAccess(counts):
    top={}
    sorted_x = sorted(counts.items(), key=operator.itemgetter(1))
    srl = len(sorted_x)
    top = sorted_x[srl-1]
    top=top[0]
    return top

def AuthorFetch(top):
    idlist=[]
    top = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', top)
    cursor.execute("SELECT * FROM referencestable where ReferenceAuthors LIKE '%"+top[0]+"%'")
    records = cursor.fetchall()
    for row in records:
        idlist.append(row[0])
        '''print "PMCID: "+str(row[0])
        print "TITLE: "+str(row[3])
        print "" '''  
    idlist = list(dict.fromkeys(idlist))
    print("Selected Important IDs:")
    print(idlist)
    return idlist

def PassAbstract(idcount,idlist):
    Abstract1 = ""
    for k in range(0,idcount):
        ids = idlist[k]
        abst = AbstractParsing(1,ids)
        Abstract1 = Abstract1+abst
    return Abstract1


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

def Summarize(Abstract):
    sentences = nltk.sent_tokenize(Abstract)
    len(sentences)
    stop_words = nltk.corpus.stopwords.words('english')
    normalize_corpus = np.vectorize(normalize_document)
    #norm_sentences = normalize_corpus(sentences)
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    tv_matrix = tv.fit_transform(sentences)
    tv_matrix = tv_matrix.toarray()
    vocab = tv.get_feature_names()
    td_matrix = tv_matrix.transpose()
    td_matrix = np.multiply(td_matrix, td_matrix > 0)
    td_matrix.shape
    num_sentences = 3
    num_topics = 2
    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
    sv_threshold = 0.5
    min_sigma_value = max(s) * sv_threshold
    s[s < min_sigma_value] = 0
    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()
    print("")
    print("")
    print("SUMMARY:")
    print('\n'.join(np.array(sentences)[top_sentence_indices]))
    
def DatabaseClose(db):
    db.close()
    

Term = TermCall()
count = SearchCall(0,0,Term) 
Count = CountConsider(count) 
IDs = SearchCall(Count,1,Term)
db = pymysql.connect("localhost","root",None,"meta_data_info")
cursor = db.cursor()
cursor.execute("USE meta_data_info")
for j in range(0,Count):
    q = str(j+1)
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print("                                             Paper Number "+q+" :")
    ids = IDs[j]
    data = MetadataCall(ids) 
    mlist = MetadataParsing(data) 
    DatabaseConnect(mlist,q,cursor,db)
    AbstractParsing(flaga,ids)  
    ReferencesParsing(ids,mlist,cursor,db) 
flaga=1
lstr = FetchAuthor(db,cursor)
counts = AuthorCount(lstr)
toplist = TopAccess(counts)
idlist = AuthorFetch(toplist)
idcount = len(idlist)
Abs = PassAbstract(idcount,idlist)
Summarize(Abs)
cursor.close()
DatabaseClose(db)
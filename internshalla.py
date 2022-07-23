import requests
from bs4 import BeautifulSoup
from datetime import datetime
startTime = datetime.now()
def jobdesc(givenUrl):
    details=""
    skills=""
    location=" "
    nexturl = "https://internshala.com"+givenUrl
    nextpage = requests.get(nexturl)
    soupnext = BeautifulSoup(nextpage.content, "html.parser")
    for p in soupnext.find_all(lambda tag: tag.name == 'div' and 
                                   tag.get('class') == ['text-container']):
        details= details+" "+p.text
    if soupnext.select_one('div:contains("Skill(s) required")') is not None:
            div=soupnext.select_one('div:contains("Skill(s) required") ~ div')
            for span in div.find_all(name="span"):
                    skills = skills+span.text+","
        
                           
    
    p = soupnext.find(name="p",attrs={'id':"location_names"})
    if p is not None:
        span = p.find(name="span")
        for a in span.find_all(name="a"):
            location = location+a.text+","
        
        
    return details,skills,location
    
job_title =[]
jobdetails =[]
jobskills =[]
joblocation=[]

for i in range(1,101):
    
    URL = "https://internshala.com/fresher-jobs/job/page-"+str(i)
    
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
 
    
    for div in soup.find_all(name="div",attrs={"class":"heading_4_5 profile"}):
        job_title.append(div.find(name='a').text)

    
    for a in soup.find_all(name="a" , attrs={"class":"view_detail_button"}):
            # print(a['href']+" end")
            skills=[]
            details=[]
            details,skills,location =jobdesc(a['href'])
            jobdetails.append(details)    
            jobskills.append(skills)
            joblocation.append(location)
          

import pandas as pd
data_frame= pd.DataFrame({'Job_title':job_title,'Job_Details': jobdetails,'Job_Skills':jobskills,'Job_location':joblocation})
data_frame = data_frame.drop(data_frame[data_frame.Job_location ==' '].index)
data_frame.to_excel('internshalla_extracted2.xlsx')
print(datetime.now()-startTime)
       

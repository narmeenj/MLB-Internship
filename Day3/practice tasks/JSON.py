import json

JsonFile="Students.json"

Students_data={
    "1001":{"Name":"Subhan","Class":"11","Degree":"Medical","Grade":"A"},
    "1002":{"Name":"Zainab","Class":"12","Degree":"Computer","Grade":"B+"}
}

with open(JsonFile,"w") as f1:
    json.dump(Students_data,f1) 

with open(JsonFile,"r") as f1:
    student_info=json.load(f1)
    
print("---- Student Data ----")  
print(json.dumps(student_info))  

updateID=input("Enter existing Student ID to update:")

if updateID in student_info:
    gradeUpdate=input("Enter New Grade for\n"+ student_info[updateID]['Name']+":")
    student_info[updateID]["Grade"]=gradeUpdate
    print("Data Updated!")

else:
    print("Student ID does not exist!")    
    
    
newID=input("\nEnter a new Student ID to add:") 
if newID in student_info:
    print("This Student ID already Exists!")
else:
    Name=input("Enter Student Name:")
    Class=input("Enter Student Class:")
    Degree=input("Enter Student Degree:") 
    Grade=input("Enter Student Grade:")  
    
    student_info[newID]={
        "Name":Name,
        "Class":Class,
        "Degree":Degree,
        "Grade":Grade    
    }
    print("Student",Name,"Added to record!")  
    
with open(JsonFile,"w") as f1:
    json.dump(student_info,f1)
print("File saved.")        
      
   
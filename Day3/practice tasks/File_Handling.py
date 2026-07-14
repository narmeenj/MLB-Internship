#Initial File Data
with open("day3_practice.txt","w")as f1:
    f1.write("This is my task for Day3.\n")
    f1.write("Which is about File Handling.\n")
    
with open("day3_practice.txt","r") as f1:
    print(f1.read())
    
with open("day3_practice.txt","a") as f1:
    f1.write("I am coding in Python.\n") 
    
#File Data after appending   
with open("day3_practice.txt","r") as f1:
    print(f1.read())
        
with open("day3_practice.txt","r") as f1:
    count=f1.readlines()  
    print("Total Number of Lines in File:",len(count))
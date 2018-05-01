from tkinter import *
import random
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim=11))
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
classifier.load_weights("loan_sanction_predict.h5")
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])


root=Tk()
root.geometry("1600x800+0+0")
root.title("Automatic Loan Sanction System")

text_Input=StringVar()
operator=""

Tops=Frame(root,width=1600,height=50,bg="powder blue",relief=SUNKEN)
Tops.pack(side=TOP)

f1=Frame(root,width=600,height=700,relief=SUNKEN,padx=20,pady=20)
f1.pack(fill="both")


f1Inner=Frame(f1,width=100,height=700,relief=SUNKEN,padx=40,pady=40,bd=10)
f1Inner.pack(fill="both",expand=True)
f1Inner.visible=True

#f2=Frame(root,width=300,height=700,bd=10,relief=SUNKEN)
#f2.pack(side=RIGHT)

#Bottoms=Frame(root,width=1600,height=50,bg="powder blue",bd=10,relief=SUNKEN,pady=50)
#Bottoms.pack(side=BOTTOM)

localtime=time.asctime(time.localtime(time.time()))


lblInfo=Label(Tops,font=('ariel',50,'bold'),text="Automatic Loan Sanction System",fg="Steel Blue",bd=10,anchor='w')
lblInfo.grid(row=0,column=0)
lblInfo=Label(Tops,font=('ariel',20,'bold'),text=localtime,fg="Steel Blue",bd=10,anchor='w')
lblInfo.grid(row=1,column=0)



LoanId=StringVar()
Gender=StringVar()
Married=StringVar()
Dependents=StringVar()
Education=StringVar()
Self_employed=StringVar()
ApplicantsIncome=StringVar()
CoApplicantsIncome=StringVar()
LoanAmount=StringVar()
LoanAmountTerms=StringVar()
CreditHistory=StringVar()
PropertyArea=StringVar()



lblID=Label(f1Inner,font=('ariel',20,'bold'),text="Loan ID",bd=16,anchor='w')
lblID.grid(row=0,column=0)
txtID=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=LoanId,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtID.grid(row=0,column=1)

lblGender=Label(f1Inner,font=('ariel',20,'bold'),text="Gender",bd=16,anchor='w')
lblGender.grid(row=1,column=0)
txtGender=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=Gender,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtGender.grid(row=1,column=1)

lblMarried=Label(f1Inner,font=('ariel',20,'bold'),text="Married",bd=16,anchor='w')
lblMarried.grid(row=2,column=0)
txtMarried=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=Married,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtMarried.grid(row=2,column=1)

lblDependents=Label(f1Inner,font=('ariel',20,'bold'),text="Dependents",bd=16,anchor='w')
lblDependents.grid(row=3,column=0)
txtDependents=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=Dependents,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtDependents.grid(row=3,column=1)

lblEducation=Label(f1Inner,font=('ariel',20,'bold'),text="Education",bd=16,anchor='w')
lblEducation.grid(row=4,column=0)
txtEducation=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=Education,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtEducation.grid(row=4,column=1)

lblSelfEmployed=Label(f1Inner,font=('ariel',20,'bold'),text="Self employed",bd=16,anchor='w')
lblSelfEmployed.grid(row=5,column=0)
txtSelfEmployed=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=Self_employed,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtSelfEmployed.grid(row=5,column=1)

lblApplicantsIncome=Label(f1Inner,font=('ariel',20,'bold'),text="Applicants Income",bd=16,anchor='w')
lblApplicantsIncome.grid(row=0,column=3)
txtApplicantsIncome=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=ApplicantsIncome,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtApplicantsIncome.grid(row=0,column=4)

lblCoApplicantsIncome=Label(f1Inner,font=('ariel',20,'bold'),text="Co Applicants Income",bd=16,anchor='w')
lblCoApplicantsIncome.grid(row=1,column=3)
txtCoApplicantsIncome=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=CoApplicantsIncome,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtCoApplicantsIncome.grid(row=1,column=4)

lblLoanAmount=Label(f1Inner,font=('ariel',20,'bold'),text="Loan Amount",bd=16,anchor='w')
lblLoanAmount.grid(row=2,column=3)
txtLoanAmount=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=LoanAmount,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtLoanAmount.grid(row=2,column=4)

lblLoanAmountTerms=Label(f1Inner,font=('ariel',20,'bold'),text="Loan Amount Terms",bd=16,anchor='w')
lblLoanAmountTerms.grid(row=3,column=3)
txtLoanAmountTerms=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=LoanAmountTerms,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtLoanAmountTerms.grid(row=3,column=4)

lblCreditHistory=Label(f1Inner,font=('ariel',20,'bold'),text="Credit History",bd=16,anchor='w')
lblCreditHistory.grid(row=4,column=3)
txtCreditHistory=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=CreditHistory,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtCreditHistory.grid(row=4,column=4)

lblPropertyArea=Label(f1Inner,font=('ariel',20,'bold'),text="Property Area",bd=16,anchor='w')
lblPropertyArea.grid(row=5,column=3)
txtPropertyArea=Entry(f1Inner,font=('ariel',20,'bold'),textvariable=PropertyArea,bd=10,insertwidth=4,bg="powder blue",justify=RIGHT)
txtPropertyArea.grid(row=5,column=4)

blankLabel=Label(f1Inner,text=" ")
blankLabel.grid(row=6,column=0)




def printStatus(f1Inner):
        LoanId=txtID.get()
        Gender=txtGender.get()
        Married=txtMarried.get()
        
        Dependents=txtDependents.get()
        Education=txtEducation.get()
        Self_employed=txtSelfEmployed.get()
        ApplicantsIncome=txtApplicantsIncome.get()
        CoApplicantsIncome=txtCoApplicantsIncome.get()
        LoanAmount=txtLoanAmount.get()
        LoanAmountTerms=txtLoanAmountTerms.get()
        CreditHistory=txtCreditHistory.get()
        PropertyArea=txtPropertyArea.get()
        
        if(Gender =="male" ):
         Gender = 1
        else :Gender = 0
        
        if(Married =="yes" ):
         Married = 1
        else :Married = 0
        
        
        if(Education =="graduate" ):
         Education = 1
        else :Education = 0
        
        if(Self_employed =="no" ):
         Self_employed = 0
        else :Self_employed = 1
        
        if(PropertyArea =="urban" ):
         PropertyArea = 1
        elif(PropertyArea =="semiurban" )  :PropertyArea = 2
        else : PropertyArea = 0
        
        y = []
        y = [Gender, Married, Dependents, Education, Self_employed, ApplicantsIncome,CoApplicantsIncome,LoanAmount,LoanAmountTerms,CreditHistory,PropertyArea]
        import numpy as np
        y = np.asarray(y)
        y = y.reshape(1,11)
        y_pred_test = classifier.predict(y)
        if y_pred_test == 1:
            x = "approved"
        else :
            x = "improve your profile and try again"


        f1Inner.pack_forget()
        status=Label(f1,text="status for : %s %s"%(LoanId,x))
        status.pack()
        	
	

submitButton=Button(f1Inner,padx=16,pady=16,bd=8,fg="black",font=('ariel',20,'bold'),text="Submit",bg="powder blue",command=lambda:printStatus(f1Inner)).grid(row=7,column=2)




root.mainloop()

from tkinter import *
from gensim.summarization import summarize

window=Tk()


label1=Label(window,text="Paste your text below and get the Summary")
label1.pack()


S = Scrollbar(window)
T = Text(window, height=20, width=100)
S.pack(side=RIGHT, fill=Y)
T.pack(side=LEFT, fill=Y)
S.config(command=T.yview)
T.config(yscrollcommand=S.set)
#T.insert(END,"paste your text here..")


def GetText():
    tex=T.get("1.0",END)
    print(tex)
    return tex

def Display_Summary():
    summary=Text_Summary()
    T.delete('1.0', END)
    T.insert(END,summary)

def Text_Summary():
    text=GetText()
    summary = summarize(text)
    return summary


b1=Button(window,text="Submit",command=Display_Summary)
b1.pack()

#GetText()

mainloop()

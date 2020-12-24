import tkinter as tk
import tkinter.font as font
from tkinter import filedialog
#from rect_noise import rect_noise
from PIL import Image, ImageTk
import predict_compact_unet
import predict_compact_fcn
#from upload_image import upload_image

window = tk.Tk()
window.title("Magnetic Resonance Imaging (MRI) Heart Segmentation")
window.iconphoto(False, tk.PhotoImage(file=r'mn.png'))
window.geometry('1080x760')


frame1 = tk.Frame(window)

label_title = tk.Label(frame1, text="Heart Segmentation Magnetic Resonance Imaging")
label_font = font.Font(size=35, weight='bold',family='Helvetica')
label_title['font'] = label_font
label_title.grid(pady=(10,10), column=2)


icon = Image.open(r'.\icons\teacher.png')
icon = icon.resize((150,150), Image.ANTIALIAS)
icon = ImageTk.PhotoImage(icon)
label_icon = tk.Label(frame1, image=icon)
label_icon.grid(row=1, pady=(5,10), column=2)

btn1_image = Image.open(r'.\icons\nn.png')
btn1_image = btn1_image.resize((50,50), Image.ANTIALIAS)
btn1_image = ImageTk.PhotoImage(btn1_image)

btn2_image = Image.open(r'.\icons\nn2.png')
btn2_image = btn2_image.resize((50,50), Image.ANTIALIAS)
btn2_image = ImageTk.PhotoImage(btn2_image)

btn5_image = Image.open(r'.\icons\exit.png')
btn5_image = btn5_image.resize((50,50), Image.ANTIALIAS)
btn5_image = ImageTk.PhotoImage(btn5_image)

btn_font = font.Font(size=15)
btn1 = tk.Button(frame1, text='MODIFIED UNET', height=90, width=280, fg='green', image=btn1_image, compound='left',command =predict_compact_unet.predict)
btn1['font'] = btn_font
btn1.grid(row=3, pady=(20,10))

btn2 = tk.Button(frame1, text='FC Network', height=90, width=280, fg='orange',command =predict_compact_fcn.predict, compound='right', image=btn2_image)
btn2['font'] = btn_font
btn2.grid(row=3, pady=(20,10), column=3, padx=(20,5))

btn5 = tk.Button(frame1, height=90, width=180, fg='red', command=window.quit, image=btn5_image)
btn5['font'] = btn_font
btn5.grid(row=6, pady=(20,10), column=2)

frame1.pack()
window.mainloop()
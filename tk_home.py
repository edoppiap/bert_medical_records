import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import ttk
import pandas as pd

'''
    This is an exploration to use tkinter as UI interface for the pretrain
'''

def center_window(window):
    # Get the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the position for the window to be centered
    x = (screen_width - window.winfo_reqwidth()) / 2
    y = (screen_height - window.winfo_reqheight()) / 2

    # Set the geometry of the window
    window.geometry("+%d+%d" % (x, y))

class Start_Window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Start window')
        self.geometry('400x300')
        center_window(self)
        
        self.pretrain_button = tk.Button(self, text='Run lm pre-train', command=self.open_lm_window)
        self.pretrain_button.pack(pady=20)
        
    def open_lm_window(self):
        self.withdraw()
        
        lm_training_window = LmTrainingWindow(self)
        lm_training_window.protocol('WM_DELETE_WINDOW', lambda: self.on_close_lm_window(lm_training_window))
        
    def on_close_lm_window(self, lm_training_window):
        lm_training_window.destroy()  # Destroy the new window
        self.deiconify()  # Show the main window again
        
class LmTrainingWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title('Pre-train')
        self.geometry('800x600')
        # self.state('zoomed')
        
        self.label = tk.Label(self, text='Here you can pre-train bert using your EHR data')
        self.label.pack(pady=20)
        
        self.selected_file_label = tk.Label(self, text='Selected file : None')
        self.selected_file_label.pack(pady=10)
        
        self.select_button = tk.Button(self, text='Select input data file', command=self.select_file)
        self.select_button.pack(pady=5)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('Spreadsheet', '*.csv'),("Text files", "*.txt")])
        if file_path and file_path.endswith('.txt'):
            text = None
            with open(file_path, 'r') as file:
                text = file.read()
                
            if text is not None: self.show_text_box(text)
        
        elif file_path and file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            
            if df is not None: 
                self.show_df(df)
                
                low_input = tk.Spinbox(self,
                                       textvariable=0,
                                       from_=0,
                                       to=len(df),
                                       increment=1)
                low_input.pack(pady=5)
                var = tk.StringVar(self)
                var.set(str(len(df)))
                high_input = tk.Spinbox(self,
                                       textvariable=var,
                                       from_=0,
                                       to=len(df),
                                       increment=1)
                high_input.pack(pady=5)
                
                
            
        if file_path:
            self.selected_file_label.config(text=f"Selected File: {file_path}")
            
    def show_text_box(self, text: str):
        frame = tk.Frame(self)
        frame.pack(fill="both", expand=False)
        
        scroll_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=20)
        scroll_text.pack(expand=False)
        scroll_text.insert(tk.END, text)
        
    def show_df(self, data: pd.DataFrame):
        frame = tk.Frame(self)
        frame.pack(fill="both", expand=False)
        
        tree = ttk.Treeview(frame)
        tree['columns'] = list(data.columns)
        tree['show'] = 'headings'
        
        for column in data.columns:
            tree.column(column, anchor=tk.CENTER, width=100)
            tree.heading(column, text=column, anchor=tk.CENTER)
        
        for i, row in data.iterrows():
            tree.insert('', tk.END, values=tuple(row))
            
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vsb.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        hsb.pack(side="bottom", fill="x")
        tree.configure(xscrollcommand=hsb.set)
        
        tree.pack(fill='both', expand=False)

if __name__ == '__main__':
    start = Start_Window()
    start.mainloop()
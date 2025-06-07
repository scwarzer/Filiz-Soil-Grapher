import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime
from matplotlib.dates import DateFormatter, DayLocator
from tkinter import Tk, filedialog, Button, Label, Frame, Entry, Canvas, Scrollbar, BooleanVar, Checkbutton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib

class VMCModel:
    def __init__(self, model_path='combined_model.joblib'):
        self.model = joblib.load(model_path)

    def predict(self, df, comp_dict, ref_dict):
        input_to_output = {
            'SC3': ('USM20', 'UST20'),
            'SC5': ('USM40', 'UST40'),
            'SC7': ('USM60', 'UST60'),
        }

        predictions = {}
        for input_col, (output_col, temp_col) in input_to_output.items():
            if input_col not in df.columns or temp_col not in df.columns:
                raise ValueError(f"Missing column: {input_col} or {temp_col}")
            comp = comp_dict.get(input_col, 0.0)
            ref_temp = ref_dict.get(input_col, df[temp_col].mean())
            corrected = df[input_col] - (df[temp_col] - ref_temp) * comp
            predictions[output_col] = self.model.predict(corrected.values.reshape(-1, 1))

        return pd.DataFrame(predictions, index=df.index)


class ControlPanel:
    def __init__(self, parent, app):
        self.app = app
        self.panel = Frame(parent, width=300)
        self.panel.pack(side="left", fill="y")
        self.panel.pack_propagate(False)

        self._build_controls()

    def _build_controls(self):
        Label(self.panel, text="Data File", font=("Arial", 10, "bold")).pack(pady=(10, 2), anchor="w", padx=10)
        Button(self.panel, text="Load Excel File", command=self.app.load_file).pack(pady=5, fill="x", padx=10)

        Label(self.panel, text="Reference Temperature", font=("Arial", 10, "bold")).pack(pady=(20, 2), anchor="w", padx=10)

        Label(self.panel, text="Temperature Correction Amount:").pack(pady=(10, 2), anchor="w", padx=10)
        self.app.comp_entry = Entry(self.panel)
        self.app.comp_entry.insert(0, "18.66")
        self.app.comp_entry.pack(fill="x", padx=10)

        for label, attr in zip(["SC3", "SC5", "SC7"], ["ref3_entry", "ref5_entry", "ref7_entry"]):
            Label(self.panel, text=f"{label} Ref Temp:").pack(pady=(5, 0), anchor="w", padx=10)
            entry = Entry(self.panel)
            entry.insert(0, "25")
            entry.pack(fill="x", padx=10)
            setattr(self.app, attr, entry)

        Button(self.panel, text="Apply Temperature Correction", command=self.app.apply_and_predict).pack(pady=(20, 10), fill="x", padx=10)
        Button(self.panel, text="Save Graph (PNG)", command=self.app.save_graph).pack(pady=(5, 10), fill="x", padx=10)

        Label(self.panel, text="Show Data", font=("Arial", 10, "bold")).pack(pady=(20, 2), anchor="w", padx=10)
        self._create_line_toggle_checkbuttons(['USM20', 'USM40', 'USM60'], ['UST20', 'UST40', 'UST60'])

        Label(self.panel, text="Read Data", font=("Arial", 10, "bold")).pack(pady=(20, 2), anchor="w", padx=10)
        self.app.hover_text = Label(self.panel, text="", justify="left", anchor="w", bg="white", font=("Consolas", 9), relief="solid", borderwidth=1)
        self.app.hover_text.pack(fill="x", padx=10)

    def _create_line_toggle_checkbuttons(self, vmc_cols, temp_cols):
        self.app.vmc_vars = {}
        for col in vmc_cols:
            var = BooleanVar(value=True)
            chk = Checkbutton(self.panel, text=col, variable=var, command=self.app.draw_plot)
            chk.pack(anchor="w", padx=20)
            self.app.vmc_vars[col] = var

        self.app.temp_vars = {}
        for col in temp_cols:
            var = BooleanVar(value=True)
            chk = Checkbutton(self.panel, text=col, variable=var, command=self.app.draw_plot)
            chk.pack(anchor="w", padx=20)
            self.app.temp_vars[col] = var

        self.app.avg_vars = {}
        for col in ['USM_AVG', 'USM_AVG_ORIGINAL']:
            var = BooleanVar(value=True)
            chk = Checkbutton(self.panel, text=col, variable=var, command=self.app.draw_plot)
            chk.pack(anchor="w", padx=20)
            self.app.avg_vars[col] = var


class GraphPanel:
    def __init__(self, parent, app):
        self.app = app
        self.frame = Frame(parent)
        self.frame.pack(side="right", fill="both", expand=True)

        self.scroll_canvas = Canvas(self.frame, highlightthickness=0)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")

        x_scroll = Scrollbar(self.frame, orient="horizontal", command=self.scroll_canvas.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll = Scrollbar(self.frame, orient="vertical", command=self.scroll_canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.scroll_canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        self.plot_frame = Frame(self.scroll_canvas)
        self.plot_window = self.scroll_canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")
        self.plot_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.app.on_hover)

        self.app.ax1 = self.ax1
        self.app.ax2 = self.ax2
        self.app.ax3 = self.ax3
        self.app.figure = self.figure
        self.app.canvas = self.canvas
        self.app.scroll_canvas = self.scroll_canvas
        self.app.plot_window = self.plot_window


class VMCApp:
    def __init__(self, root):
        self.root = root
        self.version = "1.0.0"
        self.root.title(f"Filiz Soil Grapher {self.version}")
        self.root.state("zoomed")
        self.model = VMCModel()
        self.df = None

        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        self.control_panel = ControlPanel(main_frame, self)
        self.graph_panel = GraphPanel(main_frame, self)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls")])
        if not file_path:
            return

        try:
            df = pd.read_excel(file_path)
            df['Log Date (Raw)'] = pd.to_datetime(df['Log Date (Raw)'], format="%Y-%m-%d %H:%M") + timedelta(hours=2)
            df = df.sort_values("Log Date (Raw)").reset_index(drop=True)

            df = df.rename(columns={
                'Under Soil Temperature - Filiz 1.7 - 20 cm - Data': 'UST20',
                'Under Soil Temperature - Filiz 1.7 - 40 cm': 'UST40',
                'Under Soil Temperature - Filiz 1.7 - 60 cm - Data': 'UST60'
            })

            comp_dict = {'SC3': 0.0, 'SC5': 0.0, 'SC7': 0.0}
            ref_dict = {
                'SC3': df['UST20'].mean(),
                'SC5': df['UST40'].mean(),
                'SC7': df['UST60'].mean()
            }

            original_vmc = self.model.predict(df.copy(), comp_dict, ref_dict)
            df['USM_AVG_ORIGINAL'] = original_vmc.mean(axis=1)

            self.device_id = df['DeviceId'].iloc[0] if 'DeviceId' in df.columns else "Unknown"
            self.df = df
            self.original_avg = df['USM_AVG_ORIGINAL']

            self.ref3_entry.delete(0, "end")
            self.ref3_entry.insert(0, f"{df['UST20'].mean():.2f}")
            self.ref5_entry.delete(0, "end")
            self.ref5_entry.insert(0, f"{df['UST40'].mean():.2f}")
            self.ref7_entry.delete(0, "end")
            self.ref7_entry.insert(0, f"{df['UST60'].mean():.2f}")

            self.loaded_file_path = file_path
            self.apply_and_predict()

        except Exception as e:
            print(f"Error loading file: {e}")

    def apply_and_predict(self):
        if self.df is None:
            return

        try:
            comp_val = float(self.comp_entry.get())
            comp_dict = {'SC3': comp_val, 'SC5': comp_val, 'SC7': comp_val}

            ref_dict = {
                'SC3': float(self.ref3_entry.get()),
                'SC5': float(self.ref5_entry.get()),
                'SC7': float(self.ref7_entry.get())
            }

            df_vmc = self.model.predict(self.df.copy(), comp_dict, ref_dict)

            for col in df_vmc.columns:
                self.df[col] = df_vmc[col]

            self.df['USM_AVG'] = self.df[['USM20', 'USM40', 'USM60']].mean(axis=1)
            self.draw_plot()

        except Exception as e:
            print(f"Apply error: {e}")

    def draw_plot(self):
        if self.df is None:
            return

        x = self.df['Log Date (Raw)']
        vmc_cols = ['USM20', 'USM40', 'USM60']
        temp_cols = ['UST20', 'UST40', 'UST60']

        ref3 = float(self.ref3_entry.get())
        ref5 = float(self.ref5_entry.get())
        ref7 = float(self.ref7_entry.get())
        comp_val = float(self.comp_entry.get())

        RTs = f"RT: {ref3:.1f}/{ref5:.1f}/{ref7:.1f}"
        TCAs = f"TCA: {comp_val}"

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        for key in vmc_cols:
            if self.vmc_vars[key].get():
                self.ax1.plot(x, self.df[key], label=key)
        for key in temp_cols:
            if self.temp_vars[key].get():
                self.ax2.plot(x, self.df[key], label=key)

        if self.avg_vars['USM_AVG'].get():
            self.ax3.plot(x, self.df['USM_AVG'], label='USM Average (Temp. Corrected)', color='green')
        if self.avg_vars['USM_AVG_ORIGINAL'].get() and 'USM_AVG_ORIGINAL' in self.df:
            self.ax3.plot(x, self.df['USM_AVG_ORIGINAL'], label='USM Average (Original)', color='red')

        self.ax1.set_ylabel("%VMC")
        self.ax1.set_title(f"DeviceId: {self.device_id}, Predicted %VMC with Temperature Correction ({RTs}, {TCAs})")
        self.ax1.legend()
        self.ax1.grid(True)

        self.ax2.set_ylabel("Temperature (°C)")
        self.ax2.set_title(f"DeviceId: {self.device_id}, Soil Temperatures")
        self.ax2.legend()
        self.ax2.grid(True)

        self.ax3.set_ylabel("%VMC")
        self.ax3.set_xlabel("Timestamp")
        self.ax3.set_title(f"DeviceId: {self.device_id}, Average %VMC Comparison ({RTs}, {TCAs})")
        self.ax3.legend()
        self.ax3.grid(True)
        self.ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        self.ax3.xaxis.set_major_locator(DayLocator())

        for label in self.ax3.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

        min_time = x.min().normalize()
        max_time = x.max()
        current_day = min_time

        while current_day <= max_time:
            day_start = current_day + pd.Timedelta(hours=8)
            day_end = current_day + pd.Timedelta(hours=20)
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.axvspan(current_day, day_start, color="#ccf2ff", alpha=0.3)
                ax.axvspan(day_start, day_end, color="#fff2cc", alpha=0.3)
                ax.axvspan(day_end, current_day + pd.Timedelta(days=1), color="#ccf2ff", alpha=0.3)
            current_day += pd.Timedelta(days=1)

        total_days = (max_time - min_time).days + 1
        fig_width = max(12, total_days * 1)
        self.figure.set_size_inches(fig_width, 20)
        self.scroll_canvas.itemconfig(self.plot_window, width=int(self.figure.get_size_inches()[0] * self.figure.dpi), height=int(self.figure.get_size_inches()[1] * self.figure.dpi))

        self.figure.tight_layout()
        self.canvas.draw()

    def save_graph(self):
        if self.df is None:
            return

        try:
            ref3 = float(self.ref3_entry.get())
            ref5 = float(self.ref5_entry.get())
            ref7 = float(self.ref7_entry.get())
            comp_val = float(self.comp_entry.get())

            rt = f"{ref3:.1f}-{ref5:.1f}-{ref7:.1f}"
            tca = f"{comp_val}"

            device_id = getattr(self, "device_id", "Unknown")
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_name = f"{device_id}_RT{rt}_TCA{tca}_{now_str}.png"

            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")], title="Save Graph As", initialfile=default_name)

            if file_path:
                self.figure.savefig(file_path, dpi=300)
                print(f"Graph saved to: {file_path}")

        except Exception as e:
            print(f"Error saving graph: {e}")

    def on_hover(self, event):
        if self.df is None or event.inaxes is None or event.xdata is None:
            self.hover_text.config(text="")
            return

        try:
            ax = event.inaxes
            xdata = mdates.num2date(event.xdata).replace(tzinfo=None)
            idx = (self.df['Log Date (Raw)'] - xdata).abs().idxmin()

            active_lines = []
            if ax == self.ax1:
                for col in ['USM20', 'USM40', 'USM60']:
                    if self.vmc_vars[col].get():
                        yval = self.df[col].iloc[idx]
                        active_lines.append((col, yval))
            elif ax == self.ax2:
                for col in ['UST20', 'UST40', 'UST60']:
                    if self.temp_vars[col].get():
                        yval = self.df[col].iloc[idx]
                        active_lines.append((col, yval))
            elif ax == self.ax3:
                for col in ['USM_AVG', 'USM_AVG_ORIGINAL']:
                    if col in self.df and (col in self.avg_vars and self.avg_vars[col].get()):
                        yval = self.df[col].iloc[idx]
                        active_lines.append((col, yval))

            if active_lines:
                mouse_y = event.ydata
                closest = min(active_lines, key=lambda tup: abs(tup[1] - mouse_y))
                label, yval = closest
                text = f"{label} {self.df['Log Date (Raw)'].iloc[idx].strftime('%Y-%m-%d %H:%M')} {yval:.2f}"
                self.hover_text.config(text=text)
            else:
                self.hover_text.config(text="")

        except Exception as e:
            print(f"Hover error: {e}")
            self.hover_text.config(text="")

if __name__ == "__main__":
    root = Tk()
    app = VMCApp(root)
    root.mainloop()

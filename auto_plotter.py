import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# --- 1. App Configuration ---
st.set_page_config(
    page_title="BMI Research Analytics Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Caching Function (UPDATED: SMART COLUMN DETECTION) ---
@st.cache_data
def load_and_clean_data(file):
    try:
        df = pd.read_csv(file)
        
        # --- SMART COLUMN MAPPING ---
        # 1. Create a map of {lowercase_name: original_name}
        col_map = {c.lower().strip(): c for c in df.columns}
        
        # 2. Check if we can find our targets by name (Case Insensitive)
        if 'state' in col_map and 'district' in col_map and 'bmi' in col_map:
            # Found them! Rename to standard names
            df = df.rename(columns={
                col_map['state']: 'State',
                col_map['district']: 'District',
                col_map['bmi']: 'BMI'
            })
            # Keep only relevant columns
            df = df[['State', 'District', 'BMI']]
            
        else:
            # 3. Fallback: If names don't match, assume the first 3 columns are correct
            if len(df.columns) >= 3:
                df = df.iloc[:, :3] 
                df.columns = ['State', 'District', 'BMI']
            else:
                return None, "Error: CSV must have at least 3 columns (State, District, BMI)."

        # --- Cleaning ---
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
        df.dropna(subset=['District', 'BMI'], inplace=True)
        
        return df, None
    except Exception as e:
        return None, str(e)

# --- 3. Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "research2025": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]  
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Project Password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password incorrect.", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

if check_password():
    # --- HEADER ---
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.write("# 🏥") 
    with col_title:
        st.title("BMI Research Analytics Platform")
        st.markdown("**Automated Inter-District Range Plot & Statistical Summary**")

    # --- SIDEBAR: Configuration ---
    with st.sidebar:
        st.header("1. Analysis Mode")
        analysis_mode = st.radio("Choose Mode:", ["Single Analysis", "Comparison Mode (Trend)"])
        
        st.markdown("---")
        st.header("2. Plot Settings")
        
        if analysis_mode == "Single Analysis":
            state_name = st.text_input("State Name", value="Maharashtra")
            nfhs_round = st.selectbox("NFHS Round", ["4", "5"], index=1)
            uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        else:
            state_name = st.text_input("State Name", value="Maharashtra")
            col1, col2 = st.columns(2)
            label_old = col1.text_input("Old Label", "NFHS-4")
            label_new = col2.text_input("New Label", "NFHS-5")
            
            uploaded_file_old = st.file_uploader(f"Upload {label_old} (Baseline)", type=['csv'])
            uploaded_file_new = st.file_uploader(f"Upload {label_new} (Current)", type=['csv'])

        st.markdown("---")
        x_axis_max = st.number_input("Max X-Axis Value", min_value=25, max_value=100, value=35, step=5)
        
        district_selector = st.empty()

    # =================================================================================
    # LOGIC 1: SINGLE ANALYSIS MODE
    # =================================================================================
    if analysis_mode == "Single Analysis" and uploaded_file is not None:
        df, error = load_and_clean_data(uploaded_file)
        if error: st.error(error); st.stop()

        # Highlight Logic
        all_districts = sorted(df['District'].unique())
        with st.sidebar:
            highlight_district = district_selector.selectbox("🔍 Highlight District", ["None"] + all_districts)

        tab1, tab2 = st.tabs(["📈 Visualization", "📋 Data Summary"])
        
        with tab1:
            district_medians = df.groupby('District')['BMI'].median()
            district_order = sorted(df['District'].unique()) 
            district_medians_sorted = district_medians.reindex(district_order)

            fig, ax = plt.subplots(figsize=(12, 18))
            
            # Plotting Logic
            if highlight_district == "None":
                sns.stripplot(x='BMI', y='District', data=df, order=district_order, 
                              color='#1f77b4', alpha=0.6, size=6, jitter=0.2, ax=ax)
            else:
                sns.stripplot(x='BMI', y='District', data=df, order=district_order, 
                              color='lightgrey', alpha=0.4, size=5, jitter=0.2, ax=ax)
                df_high = df[df['District'] == highlight_district]
                sns.stripplot(x='BMI', y='District', data=df_high, order=district_order, 
                              color='crimson', alpha=1.0, size=8, jitter=0.2, ax=ax)

            plt.scatter(x=district_medians_sorted.values, y=district_medians_sorted.index, 
                        marker='^', color='white', edgecolor='black', s=180, zorder=3, label='District Median')

            # Background Zones
            ymin, ymax = ax.get_ylim()
            min_val = df['BMI'].min()
            max_plot_val = float(x_axis_max)
            z1, z2, z3 = 15.0, 18.5, 25.0
            z1_c = min(z1, max_plot_val); z2_c = min(z2, max_plot_val); z3_c = min(z3, max_plot_val)

            c1=(min_val+z1_c)/2; c2=(z1_c+z2_c)/2; c3=(z2_c+z3_c)/2
            c4=(z3_c+max_plot_val)/2 if z3_c < max_plot_val else None

            if min_val < z1_c:
                ax.axvspan(min_val, z1_c, facecolor='red', alpha=0.15)
                ax.text(c1, ymax, '(<15)', ha='center', color='black', weight='bold')
            if z1_c < z2_c:
                ax.axvspan(z1_c, z2_c, facecolor='yellow', alpha=0.15)
                ax.text(c2, ymax, '(15-18.5)', ha='center', color='black', weight='bold')
            if z2_c < z3_c:
                ax.axvspan(z2_c, z3_c, facecolor='green', alpha=0.15)
                ax.text(c3, ymax, '(18.5-25)', ha='center', color='black', weight='bold')
            if z3_c < max_plot_val:
                ax.axvspan(z3_c, max_plot_val, facecolor='darkred', alpha=0.1)
                if c4: ax.text(c4, ymax, '(>25)', ha='center', color='black', weight='bold')

            ax.set_xlim(df['BMI'].min() - 1, max_plot_val)
            ax.set_title(f"Low BMI Prevalence among Adolescent Girls in {state_name} - Inter-district range plot - NFHS-{nfhs_round}", fontsize=18, pad=40)
            ax.set_xlabel("BMI(%) of Girls(15-19yrs)", fontsize=16)
            ax.set_ylabel('District', fontsize=14, fontweight='bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold'); label.set_fontsize(12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            buf = BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("📥 Download Plot", buf.getvalue(), f"{state_name}_N{nfhs_round}.png", "image/png")

        with tab2:
            st.subheader("📋 Statistical Summary")
            col1, col2 = st.columns(2)
            low_cutoff = col1.number_input("Low Cutoff", 15.0, step=0.5)
            high_cutoff = col2.number_input("High Cutoff", 25.0, step=0.5)
            
            calc_low = lambda x: (x < low_cutoff).sum() / len(x) * 100
            calc_std = lambda x: (x < 18.5).sum() / len(x) * 100
            calc_high = lambda x: (x > high_cutoff).sum() / len(x) * 100
            
            sum_df = df.groupby('District')['BMI'].agg(low=calc_low, std=calc_std, high=calc_high).reset_index().round(2)
            col_low_name = f'% BMI < {low_cutoff}'
            col_std_name = '% BMI < 18.5'
            col_high_name = f'% BMI > {high_cutoff}'
            
            sum_df.columns = ['District', col_low_name, col_std_name, col_high_name]
            
            sel_cols = st.multiselect("Columns to Display", [col_low_name, col_std_name, col_high_name], 
                                      default=[col_low_name, col_std_name, col_high_name])
            
            final_df = sum_df[['District'] + sel_cols]
            st.dataframe(final_df, use_container_width=True)
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, f"{state_name}_summary.csv", "text/csv")


    # =================================================================================
    # LOGIC 2: COMPARISON MODE (TREND)
    # =================================================================================
    elif analysis_mode == "Comparison Mode (Trend)" and uploaded_file_old and uploaded_file_new:
        df_old, err1 = load_and_clean_data(uploaded_file_old)
        df_new, err2 = load_and_clean_data(uploaded_file_new)
        
        if err1 or err2: st.error(f"Error loading files: {err1 or err2}"); st.stop()

        med_old = df_old.groupby('District')['BMI'].median()
        med_new = df_new.groupby('District')['BMI'].median()
        
        df_comp = pd.concat([med_old, med_new], axis=1, keys=['Old', 'New']).dropna()
        df_comp['Change'] = df_comp['New'] - df_comp['Old']
        df_comp = df_comp.sort_index(ascending=False) 

        district_order = sorted(df_comp.index.tolist())
        
        fig, ax = plt.subplots(figsize=(12, 18))

        df_new_filtered = df_new[df_new['District'].isin(df_comp.index)]
        sns.stripplot(x='BMI', y='District', data=df_new_filtered, order=district_order, 
                      color='lightgrey', alpha=0.3, size=4, ax=ax)

        for dist in district_order:
            val_old = df_comp.loc[dist, 'Old']
            val_new = df_comp.loc[dist, 'New']
            plt.hlines(y=dist, xmin=val_old, xmax=val_new, color='black', alpha=0.5, linewidth=1.5)

        plt.scatter(x=df_comp['Old'], y=df_comp.index, marker='^', color='grey', s=120, label=f'{label_old} Median', zorder=3)
        plt.scatter(x=df_comp['New'], y=df_comp.index, marker='^', color='green', edgecolor='black', s=160, label=f'{label_new} Median', zorder=4)

        ymin, ymax = ax.get_ylim()
        min_val = min(df_old['BMI'].min(), df_new['BMI'].min())
        max_plot_val = float(x_axis_max)
        z1, z2, z3 = 15.0, 18.5, 25.0
        z1_c = min(z1, max_plot_val); z2_c = min(z2, max_plot_val); z3_c = min(z3, max_plot_val)

        c1=(min_val+z1_c)/2; c2=(z1_c+z2_c)/2; c3=(z2_c+z3_c)/2
        c4=(z3_c+max_plot_val)/2 if z3_c < max_plot_val else None

        if min_val < z1_c:
            ax.axvspan(min_val, z1_c, facecolor='red', alpha=0.15)
            ax.text(c1, ymax, '(<15)', ha='center', color='black', weight='bold')
        if z1_c < z2_c:
            ax.axvspan(z1_c, z2_c, facecolor='yellow', alpha=0.15)
            ax.text(c2, ymax, '(15-18.5)', ha='center', color='black', weight='bold')
        if z2_c < z3_c:
            ax.axvspan(z2_c, z3_c, facecolor='green', alpha=0.15)
            ax.text(c3, ymax, '(18.5-25)', ha='center', color='black', weight='bold')
        if z3_c < max_plot_val:
            ax.axvspan(z3_c, max_plot_val, facecolor='darkred', alpha=0.1)
            if c4: ax.text(c4, ymax, '(>25)', ha='center', color='black', weight='bold')

        ax.set_xlim(min_val - 1, max_plot_val)
        ax.set_title(f"Shift in BMI: {state_name} ({label_old} vs {label_new})", fontsize=18, pad=40)
        ax.set_xlabel("BMI(%) Median Shift", fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Legend custom
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', label=f'{label_old}', markersize=10),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markeredgecolor='black', label=f'{label_new}', markersize=10),
            Line2D([0], [0], color='black', lw=1.5, label='Shift')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 Download Trend Plot", buf.getvalue(), f"{state_name}_Trend.png", "image/png")
        
        st.markdown("---")
        st.subheader("📋 Trend Analysis Table")
        st.dataframe(df_comp.round(2), use_container_width=True)

    else:
        st.info("👈 Please select a mode and upload files to begin.")
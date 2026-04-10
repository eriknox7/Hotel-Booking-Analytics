import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Analytics",
    page_icon="🏨",
    layout="wide"
)

# ── Load & Cache Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    xl = pd.ExcelFile('Datasets_Hotel_Project.xlsx')
    guests   = xl.parse('guests')
    hotels   = xl.parse('hotels')
    rooms    = xl.parse('rooms')
    bookings = xl.parse('bookings')
    services = xl.parse('services')
    feedback = xl.parse('feedback')

    df = bookings.merge(rooms[['room_id','room_type','price_per_night','hotel_id']], on='room_id', how='left')
    df = df.merge(hotels[['hotel_id','hotel_name','city','country','star_rating']], on='hotel_id', how='left')
    df = df.merge(guests[['guest_id','gender','nationality','loyalty_member','dob']], on='guest_id', how='left')

    avg_feedback = feedback.groupby(['guest_id','hotel_id'])['rating'].mean().reset_index()
    avg_feedback.rename(columns={'rating':'avg_rating'}, inplace=True)
    df = df.merge(avg_feedback, on=['guest_id','hotel_id'], how='left')

    svc_agg = services.groupby('booking_id')['service_cost'].sum().reset_index()
    svc_agg.rename(columns={'service_cost':'total_service_cost'}, inplace=True)
    df = df.merge(svc_agg, on='booking_id', how='left')
    df['total_service_cost'].fillna(0, inplace=True)

    df['stay_duration']    = (df['check_out'] - df['check_in']).dt.days
    df['revenue']          = df['total_amount'] + df['total_service_cost']
    df['customer_segment'] = pd.qcut(df['total_amount'], q=3, labels=['Budget','Mid-range','Premium'])
    df['is_cancelled']     = (df['booking_status'] == 'Cancelled').astype(int)
    df['age']              = pd.Timestamp.now().year - pd.to_datetime(df['dob']).dt.year

    df.drop_duplicates(subset='booking_id', inplace=True)
    Q1  = df['total_amount'].quantile(0.25)
    Q3  = df['total_amount'].quantile(0.75)
    IQR = Q3 - Q1
    df  = df[(df['total_amount'] >= Q1 - 1.5*IQR) &
             (df['total_amount'] <= Q3 + 1.5*IQR)].copy()
    return df

df = load_data()

# ── Sidebar Navigation ────────────────────────────────────────────
st.sidebar.title("🏨 Hotel Analytics")
page = st.sidebar.radio("Navigate to:", [
    "📊 Overview",
    "📈 Booking Trends",
    "🔗 Association Rules",
    "🤖 Classification",
    "👥 Customer Clustering",
    "💰 Revenue Prediction"
])

# ══════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🏨 Hotel Booking Analytics Dashboard")
    st.markdown("**Dataset:** 30,000 bookings | 7 sheets | July 2024 – July 2025")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bookings",   f"{len(df):,}")
    col2.metric("Total Revenue",    f"₹{df['revenue'].sum():,.0f}")
    col3.metric("Avg Stay Duration",f"{df['stay_duration'].mean():.1f} nights")
    col4.metric("Cancellation Rate",f"{df['is_cancelled'].mean()*100:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Booking Status Distribution")
        fig, ax = plt.subplots(figsize=(5,4))
        df['booking_status'].value_counts().plot(
            kind='pie', autopct='%1.1f%%', ax=ax,
            colors=['#2ecc71','#e74c3c','#3498db'])
        ax.set_ylabel('')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Revenue by Room Type")
        fig, ax = plt.subplots(figsize=(5,4))
        df.groupby('room_type')['revenue'].mean().sort_values().plot(
            kind='barh', ax=ax, color='steelblue', edgecolor='white')
        ax.set_xlabel('Avg Revenue (₹)')
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Raw Data Sample")
    st.dataframe(df[['booking_id','guest_id','hotel_name','room_type',
                      'booking_status','total_amount','stay_duration',
                      'customer_segment']].head(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2: BOOKING TRENDS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Booking Trends":
    st.title("📈 Booking Trends Over Time")

    df['checkin_month'] = pd.to_datetime(df['check_in']).dt.to_period('M')
    trend = df.groupby(['checkin_month','booking_status'])['booking_id'].count().unstack(fill_value=0)
    trend.index = trend.index.astype(str)

    fig, ax = plt.subplots(figsize=(12,5))
    trend.plot(ax=ax, linewidth=2, marker='o', markersize=4)
    ax.set_title('Booking Trends by Status')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Bookings')
    ax.grid(True, alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("Revenue Heatmap")
    col1, col2 = st.columns(2)

    with col1:
        hm1 = df.groupby(['room_type','booking_status'])['revenue'].mean().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(hm1.round(0), annot=True, fmt='.0f',
                    cmap='YlOrRd', ax=ax, linewidths=0.5)
        ax.set_title('Avg Revenue: Room × Status')
        st.pyplot(fig)
        plt.close()

    with col2:
        hm2 = df.groupby(['customer_segment','payment_status'])['revenue'].mean().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(hm2.round(0), annot=True, fmt='.0f',
                    cmap='Blues', ax=ax, linewidths=0.5)
        ax.set_title('Avg Revenue: Segment × Payment')
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════
# PAGE 3: ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    st.title("🔗 Association Rule Mining Results")

    st.info("Apriori & FP-Growth were applied on: Room Type, Customer Segment, Payment Status, Booking Status")

    col1, col2, col3 = st.columns(3)
    col1.metric("Frequent Itemsets", "76")
    col2.metric("Rules Generated",   "60")
    col3.metric("Min Support",       "0.05")

    st.divider()
    st.subheader("Top Association Rules")

    rules_data = {
        'Antecedent'  : ['Room_Double','Status_No Show','Payment_Paid',
                          'Segment_Budget','Room_Suite','Payment_Refunded',
                          'Segment_Premium','Payment_Pending'],
        'Consequent'  : ['Status_No Show','Room_Double','Room_Double',
                          'Status_No Show','Segment_Budget','Room_Suite',
                          'Status_Cancelled','Status_Cancelled'],
        'Support'     : [0.0896,0.0896,0.0896,0.1153,0.0795,0.0786,0.1105,0.1096],
        'Confidence'  : [0.3477,0.2652,0.2649,0.3457,0.3396,0.2384,0.3316,0.3300],
        'Lift'        : [1.0296,1.0296,1.0283,1.0239,1.0187,1.0181,1.0089,1.0041]
    }
    rules_df = pd.DataFrame(rules_data)

    min_lift = st.slider("Filter by minimum Lift:", 1.0, 1.05, 1.0, 0.001)
    filtered = rules_df[rules_df['Lift'] >= min_lift]
    st.dataframe(filtered, use_container_width=True)

    st.divider()
    st.subheader("Key Insights")
    st.success("✅ Double rooms are associated with No Show behaviour (Lift: 1.03)")
    st.warning("⚠️ Premium segment & Suite/Deluxe rooms show higher cancellation tendency")
    st.info("💡 Payment_Pending bookings are more likely to result in cancellations")

# ══════════════════════════════════════════════════════════════════
# PAGE 4: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Classification":
    st.title("🤖 Booking Status Classification")

    @st.cache_resource
    def train_classifier(df):
        features = ['room_type','total_amount','customer_segment',
                    'stay_duration','payment_status']
        target   = 'booking_status'
        mdf = df[features + [target]].dropna().copy()
        for col in ['room_type','customer_segment','payment_status','booking_status']:
            mdf[col] = LabelEncoder().fit_transform(mdf[col].astype(str))
        X = mdf[features]
        y = mdf[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        return rf, X_test, y_test, features

    rf_model, X_test, y_test, features = train_classifier(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model",  "Random Forest")
    col2.metric("Accuracy",    "~33%")
    col3.metric("Note",        "Balanced classes")
    col4.metric("Features",    str(len(features)))

    st.divider()
    st.subheader("Feature Importance")
    imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8,4))
    imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Feature Importance — Random Forest')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("Model Performance Summary")
    perf = pd.DataFrame({
        'Model'    : ['Random Forest','SVM','Logistic Regression','Decision Tree'],
        'Accuracy' : [0.364, 0.362, 0.332, 0.328],
        'F1 Score' : [0.367, 0.334, 0.301, 0.327]
    })
    st.dataframe(perf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5: CLUSTERING
# ══════════════════════════════════════════════════════════════════
elif page == "👥 Customer Clustering":
    st.title("👥 Customer Segmentation")

    @st.cache_data
    def get_clusters(df):
        booking_freq = df.groupby('guest_id')['booking_id'].count().reset_index()
        booking_freq.columns = ['guest_id','booking_frequency']
        cust_df = df.groupby('guest_id').agg(
            total_spending    = ('total_amount','sum'),
            avg_stay          = ('stay_duration','mean'),
            total_services    = ('total_service_cost','sum'),
            cancellation_rate = ('is_cancelled','mean')
        ).reset_index()
        cust_df = cust_df.merge(booking_freq, on='guest_id', how='left')
        X = cust_df[['total_spending','avg_stay','total_services',
                     'cancellation_rate','booking_frequency']]
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        cust_df['cluster'] = km.fit_predict(Xs)
        label_map = cust_df.groupby('cluster')['total_spending'].mean().rank().map(
            {1:'Budget', 2:'Mid-range', 3:'Premium'})
        cust_df['segment'] = cust_df['cluster'].map(label_map)
        return cust_df

    cust_df = get_clusters(df)

    col1, col2, col3 = st.columns(3)
    for seg, col, color in zip(
        ['Premium','Mid-range','Budget'],
        [col1, col2, col3],
        ['🥇','🥈','🥉']
    ):
        sub = cust_df[cust_df['segment'] == seg]
        col.metric(f"{color} {seg}",
                   f"{len(sub):,} customers",
                   f"Avg ₹{sub['total_spending'].mean():,.0f}")

    st.divider()
    st.subheader("Cluster Profiles")
    profile = cust_df.groupby('segment')[
        ['total_spending','avg_stay','total_services',
         'cancellation_rate','booking_frequency']
    ].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending by Segment")
        fig, ax = plt.subplots(figsize=(5,4))
        cust_df.groupby('segment')['total_spending'].mean().sort_values().plot(
            kind='bar', ax=ax, color=['#e74c3c','#f39c12','#2ecc71'],
            edgecolor='white')
        ax.set_ylabel('Avg Total Spending (₹)')
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Cancellation Rate by Segment")
        fig, ax = plt.subplots(figsize=(5,4))
        cust_df.groupby('segment')['cancellation_rate'].mean().sort_values().plot(
            kind='bar', ax=ax, color=['#2ecc71','#f39c12','#e74c3c'],
            edgecolor='white')
        ax.set_ylabel('Avg Cancellation Rate')
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════
# PAGE 6: REVENUE PREDICTION
# ══════════════════════════════════════════════════════════════════
elif page == "💰 Revenue Prediction":
    st.title("💰 Revenue Prediction (ANN)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model",  "ANN (Feedforward)")
    col2.metric("R² Score","0.9987")
    col3.metric("RMSE",   "₹31.20")

    st.divider()
    st.subheader("🔮 Predict Revenue for a New Booking")

    col1, col2 = st.columns(2)
    with col1:
        room_type     = st.selectbox("Room Type", ['Single','Double','Suite','Deluxe'])
        stay_duration = st.slider("Stay Duration (nights)", 1, 30, 5)
        star_rating   = st.slider("Hotel Star Rating", 1, 5, 3)
    with col2:
        payment_status  = st.selectbox("Payment Status", ['Paid','Pending','Refunded'])
        loyalty_member  = st.selectbox("Loyalty Member", [0, 1])
        is_cancelled    = st.selectbox("Booking Cancelled?", [0, 1])

    # Simple rule-based estimate (proxy for ANN)
    base = {'Single': 800, 'Double': 1000, 'Suite': 1400, 'Deluxe': 1200}
    est  = base[room_type] + (stay_duration * 50) + (star_rating * 30)
    if loyalty_member: est *= 1.1
    if is_cancelled:   est *= 0.6

    st.divider()
    st.success(f"### 💵 Estimated Revenue: ₹{est:,.0f}")
    st.caption("Note: This is a rule-based estimate for demonstration. "
               "The full ANN model achieves R²=0.9987 on the training data.")

    st.divider()
    st.subheader("Model Comparison")
    comp = pd.DataFrame({
        'Model'  : ['Linear Regression','Random Forest','Decision Tree','ANN'],
        'RMSE'   : ['~0','5.8','12.1','31.2'],
        'R²'     : ['~1.0','~1.0','~1.0','0.9987'],
        'Winner' : ['🥇','🥈','🥉','4th']
    })
    st.dataframe(comp, use_container_width=True)
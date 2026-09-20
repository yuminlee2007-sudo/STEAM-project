import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor

# --- 1. 페이지 기본 설정 ---
st.set_page_config(
    page_title="Hà Đông AI Traffic Predictor",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 하동(Hà Đông) AI 실시간 & 미래 교통 혼잡 예측 시스템")
st.markdown("과거 시계열 데이터와 머신러닝(Random Forest) 기반으로 **하동구 전체 주요 도로**의 미래 교통량을 예측합니다.")

# --- 2. 하동구 주요 5대 도로 좌표 정의 ---
HA_DONG_ROADS = {
    "Trần Phú - Quang Trung (중심 축)": {
        "coords": [[20.9830, 105.7920], [20.9730, 105.7790], [20.9630, 105.7650]],
        "base_congestion": 80
    },
    "Tố Hữu (북부 신도시 축)": {
        "coords": [[20.9890, 105.7860], [20.9780, 105.7680], [20.9690, 105.7500]],
        "base_congestion": 75
    },
    "Lê Trọng Tấn (서부 외곽 순환 축)": {
        "coords": [[20.9750, 105.7480], [20.9600, 105.7580], [20.9480, 105.7680]],
        "base_congestion": 50
    },
    "Phùng Hưng (동부 병원/주거 축)": {
        "coords": [[20.9800, 105.7950], [20.9680, 105.7910], [20.9550, 105.7930]],
        "base_congestion": 65
    },
    "Văn Khê - Nguyễn Thanh Bình (내부 연결 축)": {
        "coords": [[20.9720, 105.7650], [20.9620, 105.7720]],
        "base_congestion": 45
    }
}

# --- 3. AI 예측 모델 학습 ---
@st.cache_resource
def train_traffic_ai():
    hours = np.tile(np.arange(24), 7)
    days = np.repeat(np.arange(7), 24)
    rush_hour_weight = np.where((hours >= 7) & (hours <= 9), 35, 0) + \
                       np.where((hours >= 17) & (hours <= 19), 40, 0)
    congestion_data = 20 + rush_hour_weight + np.random.normal(0, 5, len(hours))
    congestion_data = np.clip(congestion_data, 10, 100)
    
    X = np.column_stack((hours, days))
    y = congestion_data
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

ai_model = train_traffic_ai()

# --- 4. 사이드바 UI ---
st.sidebar.header("⚙️ 예측 설정 옵션")
selected_time_offset = st.sidebar.select_slider(
    "예측 시점 선택",
    options=[0, 15, 30, 45, 60],
    value=30,
    format_func=lambda x: "현재 실시간" if x == 0 else f"{x}분 후 예측"
)

weather = st.sidebar.selectbox("날씨 조건", ["맑음 (Clear)", "비/폭우 (Rain)", "안개 (Fog)"])
weather_factor = 1.25 if "비" in weather else 1.0

# --- 5. 예측 연산 ---
now = datetime.datetime.now()
target_time = now + datetime.timedelta(minutes=selected_time_offset)
target_hour = target_time.hour
target_day = target_time.weekday()

base_predicted_score = ai_model.predict([[target_hour, target_day]])[0] * weather_factor

# --- 6. 대시보드 레이아웃 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📍 하동구 전체 교통 지적도 ({target_time.strftime('%H:%M')} 기준)")
    m = folium.Map(location=[20.9712, 105.7766], zoom_start=13, tiles="cartodbpositron")
    
    road_summary = []
    
    for road_name, data in HA_DONG_ROADS.items():
        road_score = min(100, int((base_predicted_score * (data["base_congestion"] / 60))))
        
        if road_score < 45:
            color = "#2ecc71"
            status = "🟢 원활"
            speed = f"{np.random.randint(40, 50)} km/h"
        elif road_score < 75:
            color = "#f39c12"
            status = "🟡 서행"
            speed = f"{np.random.randint(20, 35)} km/h"
        else:
            color = "#e74c3c"
            status = "🔴 혼잡"
            speed = f"{np.random.randint(8, 15)} km/h"
            
        road_summary.append({
            "도로명": road_name,
            "예측 혼잡도": f"{road_score} / 100",
            "상태": status,
            "예상 평균 속도": speed
        })
        
        folium.PolyLine(
            locations=data["coords"],
            color=color,
            weight=8,
            opacity=0.8,
            popup=f"<b>{road_name}</b><br>예측 상태: {status}<br>혼잡지수: {road_score}/100"
        ).add_to(m)

    st_folium(m, width=800, height=500)

with col2:
    st.subheader("📊 주요 도로별 예측 리포트")
    st.write(f"**예측 기준 시각:** {target_time.strftime('%Y-%m-%d %H:%M')}")
    
    summary_df = pd.DataFrame(road_summary)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    st.info("""
    💡 **AI 분석 인사이트:**
    * **Trần Phú - Quang Trung** 도로는 출퇴근 시간 정체 지수가 가장 높습니다.
    * 비가 올 경우 전체 혼잡 지수가 **25% 상승**하도록 모델링되었습니다.
    * 30분 후 혼잡도가 높을 경우 지상철(Cát Linh - Hà Đông) 이용이 권장됩니다.
    """)

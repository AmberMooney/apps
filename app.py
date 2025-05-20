import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium

#import torch
#from models.GLSTM import GLSTM_Regressor

#url = ""  # OSF direct download link
#response = requests.get(url)
#df = pd.read_csv(io.StringIO(response.text))

# Initialize and load the model, align new data, preprocess, format for input, instantiate model, pass through model, post process outputs.. 

#def process_evaluate_test(eval_year, df, target_col=target_col):
#    years = sorted(inputs['Year'].unique())
#    train_years = years[2002:2006] 
#    train_df = inputs[inputs['Year'].isin(train_years)]
#    
#    train_df = inputs[inputs['Year'] < eval_year]
#    train_dataset = SpatioTemporalDataset(train_df, features)
#    test_df = inputs[inputs['Year'] == eval_year]
#    test_dataset = SpatioTemporalDataset(test_df, features)
#    
#    scaler = train_dataset.normalize()
#    test_dataset.tensor = torch.tensor(
#        scaler.transform(test_dataset.tensor.view(-1, train_dataset.num_features).numpy())
#        ).view(test_dataset.T, test_dataset.N, train_dataset.num_features)
#
#    y_test_raw = test_df[[target_col]].values
#
#    y_test_log = np.log1p(y_test_raw)
    
#    X_test_scaled = test_dataset.tensor    # [T, N, F]
    
#    data_test = Data(
#        x=X_test_scaled,
#        y=torch.tensor(y_test_log, dtype=torch.float32),
#        edge_index=edge_index
#    )
#    input_dim = test_dataset.num_features
#    model = GLSTM_Regressor(
#        in_channels=input_dim,
#        gnn_hidden=64,
#        lstm_hidden=64,
#        out_channels=1,  # for scalar regression
#        gnn_type="gat"
#    )
#    print("Processing Test Done")
#    model.load_state_dict(torch.load('model_fold1.pth', weights_only=True))
#    model.eval()
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#    #criterion = torch.nn.MSELoss()
#    print("Model Instatiation Done")
#    test_loss, mae_test, rmse_test, test_out, test_mae_per_epoch, test_rmse_per_epoch, mae_nonzero, exact, soft = evaluate(model, data_test, alpha=5.0)
#    print("Model Call Done")
#    tensor1_flat = test_out.squeeze(-1).reshape(-1).cpu().numpy()
#    tensor2_flat = data_test.y.reshape(-1).cpu().numpy()
#    tensor_order = pd.MultiIndex.from_product([test_dataset.time_keys, test_dataset.fips_list], names=["Weeks", "FIPS"]).to_frame(index=False)
#    results_df = tensor_order.copy()
#    results_df['Preds'] = np.expm1(tensor1_flat)
#    results_df['Target'] = np.expm1(tensor2_flat)
#    print("Post Processing Done - Ready to plot!")

#    return results_df

# Map predictions back to hexes



st.set_page_config(layout="wide")
st.title("Screwworm Risk Dashboard")

# Load Data
gdf = gpd.read_file("sample_hex.geojson")
weather_df = pd.read_csv("weather_forecast.csv")

# Setup Tabs 
tab1, tab2, tab3 = st.tabs([" Risk Map", " Upload Inputs", " Model Output"])

# Store uploaded data 
new_weather = None
new_drops = None
gdf_updated = gdf.copy()
# Tab1: Risk map
with tab1:
    st.header(" Risk Map")
    st.markdown("Each node is colored by its risk score. This can change based on uploaded inputs.")

    #if new_weather is not None and new_drops is not None:
    #gdf_updated = run_model_prediction(new_weather, new_drops, gdf)


    # Simulate model prediction based on uploads
    def simulate_model(weather, drops, gdf):
        import numpy as np
        gdf = gdf.copy()
        gdf["risk_score"] = np.clip(0.2 + 0.01 * np.random.randn(len(gdf)), 0, 1)
        return gdf

    if "uploaded_weather" in st.session_state and "uploaded_drops" in st.session_state:
        gdf_updated = simulate_model(
            st.session_state["uploaded_weather"],
            st.session_state["uploaded_drops"],
            gdf
        )

    # Plot
    m = folium.Map(location=[7.8, -77.6], zoom_start=7)
    folium.Choropleth(
        geo_data=gdf_updated,
        data=gdf_updated,
        columns=["id", "risk_score"],
        key_on="feature.properties.id",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.5,
        legend_name="Risk Score"
    ).add_to(m)

    st_folium(m, width=1000, height=500)


# TAB2: inputs
with tab2:
    st.header("Upload New Inputs")

    weather_file = st.file_uploader("Upload New Weather Data (CSV)", type=["csv"])
    drops_file = st.file_uploader("Upload New SIT Drop Data (CSV)", type=["csv"])

    if weather_file is not None:
        new_weather = pd.read_csv(weather_file)
        st.session_state["uploaded_weather"] = new_weather
        st.success("Weather data uploaded!")
        st.dataframe(new_weather)

    if drops_file is not None:
        new_drops = pd.read_csv(drops_file)
        st.session_state["uploaded_drops"] = new_drops
        st.success("SIT drop data uploaded!")
        st.dataframe(new_drops)


#TAB3: Model Output
with tab3:
    st.header(" Simulated Model Output")
    st.dataframe(gdf_updated[["id", "risk_score"]])

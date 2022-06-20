from Imports import *

cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal; text-align: center;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #01153E; color: white; text-align: center;',
}


def mat_vec(A, y):
    return np.dot(A, y)

def L_SI(y, angles):

        phi, theta, psi = angles
        
        Rx = np.array([[1, 0, 0],
                      [0, float(np.cos(phi)), float(np.sin(phi))],
                      [0, -float(np.sin(phi)), float(np.cos(phi))]])
        Ry = np.array([[float(np.cos(theta)), 0, -float(np.sin(theta))],
                      [0, 1, 0],
                      [float(np.sin(theta)), 0, float(np.cos(theta))]])
        Rz = np.array([[float(np.cos(psi)), float(np.sin(psi)), 0],
                      [-float(np.sin(psi)), float(np.cos(psi)), 0],
                      [0, 0, 1]])

        L_SI = np.dot(Rz.T, np.dot(Ry.T, Rx.T))
        
        y_new = np.zeros(y.shape)
        
        for i in range(len(y[0])):
            
            y_new[:,i] = mat_vec(L_SI, y[:,i])
        
        return y_new
    
def actuator_length(angles, u_s, b_i, r_si):
    
    new_r_si = np.array([[r_si[0]], [r_si[1]], [-r_si[2]]])
    u_i = L_SI(u_s, angles) + new_r_si.repeat(len(u_s[0]), axis=0).reshape(-1,len(u_s[0]))
    
    return [np.linalg.norm(u_i[:,i] - b_i[:,i]) for i in range(len(u_s[0]))]

def plot_frame_new(u_s, b_i, r_si_0, ax, scale=1, u_s_updated=[], linewidths=[2,3]):
    u_s_ = u_s * scale
    b_i_ = b_i * scale
    r_si_0_ = r_si_0 * scale
    
    if type(u_s_updated) != list:
        u_s_updated_ = u_s_updated * scale
    else:
        u_s_updated_ = u_s * scale
        
    if type(u_s_updated) != list:
        X_top = (np.append(u_s_updated_[0], u_s_updated_[0][0]) + r_si_0_[0]).reshape(-1,1)
        Y_top = (np.append(u_s_updated_[1], u_s_updated_[1][0]) + r_si_0_[1]).reshape(-1,1)
        Z_top = (np.append(u_s_updated_[2], u_s_updated_[2][0]) - r_si_0_[2]).reshape(-1,1)
    else:
        X_top = (np.append(u_s_[0], u_s_[0][0]) + r_si_0_[0]).reshape(-1,1)
        Y_top = (np.append(u_s_[1], u_s_[1][0]) + r_si_0_[1]).reshape(-1,1)
        Z_top = (np.append(u_s_[2], u_s_[2][0]) - r_si_0_[2]).reshape(-1,1)

    X_bottom = np.append(b_i_[0], b_i_[0][0]).reshape(-1,1)
    Y_bottom = np.append(b_i_[1], b_i_[1][0]).reshape(-1,1)
    Z_bottom = np.append(b_i_[2], b_i_[2][0]).reshape(-1,1)

#     ax.plot_wireframe(X_bottom, Y_bottom, Z_bottom, color='darkslategrey', linewidth=5)

    pc = Poly3DCollection([list(zip(X_top.reshape(-1,),Y_top.reshape(-1,),Z_top.reshape(-1,)))], edgecolors='darkslategrey', linewidths=linewidths[0], alpha=0.5)
    pc.set_facecolor('aquamarine')
    ax.add_collection3d(pc) 
    
    pc = Poly3DCollection([list(zip(X_bottom.reshape(-1,),Y_bottom.reshape(-1,),Z_bottom.reshape(-1,)))], edgecolors='darkslategrey', linewidths=linewidths[0], alpha=0.5)
    pc.set_facecolor('lightcyan')
    ax.add_collection3d(pc) 
    
    for i in range(6):
        X_actuator = np.array([float(u_s_updated_[0][i]+r_si_0_[0]), b_i_[0][i]]).reshape(-1,1)
        Y_actuator = np.array([float(u_s_updated_[1][i]+r_si_0_[1]), b_i_[1][i]]).reshape(-1,1)
        Z_actuator = np.array([float(u_s_updated_[2][i]-r_si_0_[2]), b_i_[2][i]]).reshape(-1,1)
        length = np.linalg.norm(np.array([X_actuator[0], Y_actuator[0], Z_actuator[0]]) - np.array([X_actuator[1], Y_actuator[1], Z_actuator[1]])) 
        if length > 110 or length < 100:
            color = 'gold'
            if length > 115 or length < 95:
                color = 'orange'
                if length > 121 or length < 91:
                    color = 'red'
        else:
            color = 'midnightblue'
        ax.plot_wireframe(X_actuator, Y_actuator, Z_actuator, linestyle='solid', color=color, linewidth=linewidths[1])
    
    return ax, plt

with st.sidebar:
    selected = option_menu("Stewart Platform Kinematic Simulator", ["Home", 'Settings', 'Geometry', 'Results', 'Animation'], 
        icons=['house-fill', 'sliders', 'layout-wtf', 'bar-chart-fill', 'joystick'], menu_icon="cast", default_index=1)
#     selected
        
act_default_upper_x = np.array([-74.85, 33.30, 41.55, 41.55, 33.30, -74.85])
act_default_upper_y = np.array([-4.78, -67.21, -62.43, 62.43, 67.21, 4.78])
act_default_upper_z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

act_default_lower_x = np.array([-36.78, -29.08, 64.0, 64.0, -29.08, -36.78])
act_default_lower_y = np.array([-54.61, -59.26, -4.45, 4.45, 59.26, 54.61])
act_default_lower_z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

other_default = np.array([86.18, 91.44, 121.92, 0.0, 0.0, 0.0])
    
dict_geo = {'act1_upper_x':act_default_upper_x[0], 'act2_upper_x':act_default_upper_x[1], 'act3_upper_x':act_default_upper_x[2], 'act4_upper_x':act_default_upper_x[3], 'act5_upper_x':act_default_upper_x[4], 'act6_upper_x':act_default_upper_x[5],
    'act1_upper_y':act_default_upper_y[0], 'act2_upper_y':act_default_upper_y[1], 'act3_upper_y':act_default_upper_y[2], 'act4_upper_y':act_default_upper_y[3], 'act5_upper_y':act_default_upper_y[4], 'act6_upper_y':act_default_upper_y[5],
    'act1_upper_z':act_default_upper_z[0], 'act2_upper_z':act_default_upper_z[1], 'act3_upper_z':act_default_upper_z[2], 'act4_upper_z':act_default_upper_z[3], 'act5_upper_z':act_default_upper_z[4], 'act6_upper_z':act_default_upper_z[5],
    'act1_lower_x':act_default_lower_x[0], 'act2_lower_x':act_default_lower_x[1], 'act3_lower_x':act_default_lower_x[2], 'act4_lower_x':act_default_lower_x[3], 'act5_lower_x':act_default_lower_x[4], 'act6_lower_x':act_default_lower_x[5],
    'act1_lower_y':act_default_lower_y[0], 'act2_lower_y':act_default_lower_y[1], 'act3_lower_y':act_default_lower_y[2], 'act4_lower_y':act_default_lower_y[3], 'act5_lower_y':act_default_lower_y[4], 'act6_lower_y':act_default_lower_y[5],
    'act1_lower_z':act_default_lower_z[0], 'act2_lower_z':act_default_lower_z[1], 'act3_lower_z':act_default_lower_z[2], 'act4_lower_z':act_default_lower_z[3], 'act5_lower_z':act_default_lower_z[4], 'act6_lower_z':act_default_lower_z[5],
    'pivot_x':other_default[3], 'pivot_y':other_default[4], 'pivot_z':other_default[5], 'minL':other_default[1], 'maxL':other_default[2], 'dist_frame':other_default[0]}

for key, value in dict_geo.items():
    if key+'_new' not in st.session_state:
        st.session_state[key+'_new'] = value
#     if key not in st.session_state:
#         st.session_state[key] = value

for key, value in dict_geo.items():
    if (selected != 'Geometry') or (key not in st.session_state):
        st.session_state[key] = st.session_state[key+'_new']
    
if 'unit' not in st.session_state:
    st.session_state.unit = 1.0
    st.session_state.disp_unit_flag = 'Centimeters'
    st.session_state.disp_unit = 'Centimeters'
if 'unit_' not in st.session_state:
    st.session_state.unit_ = 1.0
    st.session_state.rot_unit_flag = 'Degrees'
    st.session_state.rot_unit = 'Degrees'
    
if 'disp_unit_new' not in st.session_state:
    st.session_state.disp_unit_new = 'Centimeters'

# if 'disp_unit' not in st.session_state:
# #     st.session_state.disp_unit = 'Centimeters
#     st.session_state.disp_unit = st.session_state.disp_unit_new

if 'rot_unit_new' not in st.session_state:
    st.session_state.rot_unit_new = 'Degrees'

# if 'rot_unit' not in st.session_state:
# #     st.session_state.rot_unit = 'Degrees'
#     st.session_state.rot_unit = st.session_state.rot_unit_new

if selected != 'Settings':
    st.session_state.disp_unit = st.session_state.disp_unit_new
    st.session_state.rot_unit = st.session_state.rot_unit_new

for item in ['surge_move', 'sway_move', 'heave_move', 'roll_move', 'pitch_move', 'yaw_move']:
    if item not in st.session_state:
        st.session_state[item] = 0.0
    
def set_geometry():
    
    st.session_state.update({val+'_new': st.session_state[val] for val in dict_geo.keys()})
    for val in dict_geo.keys():
        if st.session_state[val+'_new'] != st.session_state[val]:
            st.session_state[val] = st.session_state[val+'_new']
    st.session_state.unit = 1.0
    
#     new_title = '<p style="font-family:sans-serif; color:Green; font-size: 14px; position: revert;">Changes are successfully saved!</p>'

def reset_geo():
    st.session_state.update({val+'_new': item/st.session_state.unit for val, item in dict_geo.items()})
    st.session_state.update({val: item/st.session_state.unit for val, item in dict_geo.items()})
    
with st.sidebar:
    ####
    angles = np.array([[0.0], [0.0], [0.0]])
    linears = np.array([[0.0], [0.0], [0.0]])
    angles_0 = angles.copy()
    
    b_i = np.array([[st.session_state['act1_lower_x_new'], 
                     st.session_state['act2_lower_x_new'], 
                     st.session_state['act3_lower_x_new'], 
                     st.session_state['act4_lower_x_new'], 
                     st.session_state['act5_lower_x_new'], 
                     st.session_state['act6_lower_x_new']],
                    [st.session_state['act1_lower_y_new'], 
                     st.session_state['act2_lower_y_new'], 
                     st.session_state['act3_lower_y_new'], 
                     st.session_state['act4_lower_y_new'], 
                     st.session_state['act5_lower_y_new'], 
                     st.session_state['act6_lower_y_new']],
                    [st.session_state['act1_lower_z_new'], 
                     st.session_state['act2_lower_z_new'], 
                     st.session_state['act3_lower_z_new'], 
                     st.session_state['act4_lower_z_new'], 
                     st.session_state['act5_lower_z_new'], 
                     st.session_state['act6_lower_z_new']]])

    u_s = np.array([[st.session_state['act1_upper_x_new'], 
                     st.session_state['act2_upper_x_new'], 
                     st.session_state['act3_upper_x_new'], 
                     st.session_state['act4_upper_x_new'], 
                     st.session_state['act5_upper_x_new'], 
                     st.session_state['act6_upper_x_new']],
                    [st.session_state['act1_upper_y_new'], 
                     st.session_state['act2_upper_y_new'], 
                     st.session_state['act3_upper_y_new'], 
                     st.session_state['act4_upper_y_new'], 
                     st.session_state['act5_upper_y_new'], 
                     st.session_state['act6_upper_y_new']],
                    [st.session_state['act1_upper_z_new'], 
                     st.session_state['act2_upper_z_new'], 
                     st.session_state['act3_upper_z_new'], 
                     st.session_state['act4_upper_z_new'], 
                     st.session_state['act5_upper_z_new'], 
                     st.session_state['act6_upper_z_new']]])
    
    r_si_0 = np.array([[0], [0], [-st.session_state.dist_frame_new]])
    r_si = r_si_0 + linears
    fig = plt.figure(figsize=(5,5))
    ax = plt.axes(projection='3d')
#     ax._axis3don = False
    
    plot3d_x_lim_min = round(min(min(u_s[0,:])+r_si_0[0][0],min(b_i[0,:])),2)*1.15
    plot3d_x_lim_max = round(max(max(u_s[0,:])+r_si_0[0][0],max(b_i[0,:])),2)*1.15
    plot3d_y_lim_min = round(min(min(u_s[1,:])+r_si_0[1][0],min(b_i[1,:])),2)*1.15
    plot3d_y_lim_max = round(max(max(u_s[1,:])+r_si_0[1][0],max(b_i[1,:])),2)*1.15
    plot3d_z_lim_min = round(max(b_i[2,:]),2)*1.12
    plot3d_z_lim_max = round(-(min(u_s[2,:])+r_si_0[2][0]),2)*1.12
    
    ax.set_xlim([plot3d_x_lim_min, plot3d_x_lim_max]) 
    ax.set_ylim([plot3d_y_lim_min, plot3d_y_lim_max]) 
    ax.set_zlim([plot3d_z_lim_min, plot3d_z_lim_max])
    
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelbottom=False)
    ax.zaxis.set_tick_params(labelbottom=False)
        
    ax_, plot_ = plot_frame_new(u_s=u_s, b_i=b_i, r_si_0=r_si_0, ax=ax, scale=1, linewidths=[1,1.5])
    
    ax_.scatter3D(st.session_state.pivot_x_new, st.session_state.pivot_y_new, st.session_state.dist_frame_new + st.session_state.pivot_z_new, marker ='+', color='red', s=200)

    st.pyplot(fig)
    ####
    
    uploaded_file = st.file_uploader("Choose a .XLSX file", accept_multiple_files=False)
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)        
        ind_start_upper = df.index[(pd.Series((df.iloc[i,0] == 'UPPER FRAME' and df.iloc[i+1,0] == 'X')  for i in df.index))][0] + 2
        ind_start_lower = df.index[(pd.Series((df.iloc[i,0] == 'LOWER FRAME' and df.iloc[i+1,0] == 'X')  for i in df.index))][0] + 2
        ind_start_actuator = df.index[(pd.Series((df.iloc[i,0] == 'ACTUATOR' and df.iloc[i+1,0] == 'MIN')  for i in df.index))][0] + 2
        
        df_upper = df.iloc[ind_start_upper:ind_start_upper+6,:].reset_index(drop=True)
        df_upper = df_upper.rename(columns={df_upper.columns[0]:'X', df_upper.columns[1]:'Y', df_upper.columns[2]:'Z'})
        df_lower = df.iloc[ind_start_lower:ind_start_lower+6,:].reset_index(drop=True)
        df_lower = df_lower.rename(columns={df_lower.columns[0]:'X', df_lower.columns[1]:'Y', df_lower.columns[2]:'Z'})
        df_actuator = df.iloc[ind_start_actuator:ind_start_actuator+1,:].reset_index(drop=True)
        df_actuator = df_actuator.rename(columns={df_actuator.columns[0]:'MIN', df_actuator.columns[1]:'MAX', df_actuator.columns[2]:'FRAME DIST'})
        st.write("UPPER FRAME")
        st.dataframe(df_upper)
        st.write("LOWER FRAME")
        st.dataframe(df_lower)
        st.write("ACTUATOR")
        st.dataframe(df_actuator)
        
        st.session_state['act1_upper_x_new'] = df_upper.iloc[0][0]
        st.session_state['act2_upper_x_new'] = df_upper.iloc[1][0]
        st.session_state['act3_upper_x_new'] = df_upper.iloc[2][0]
        st.session_state['act4_upper_x_new'] = df_upper.iloc[3][0]
        st.session_state['act5_upper_x_new'] = df_upper.iloc[4][0]
        st.session_state['act6_upper_x_new'] = df_upper.iloc[5][0]
        
        st.session_state['act1_upper_y_new'] = df_upper.iloc[0][1]
        st.session_state['act2_upper_y_new'] = df_upper.iloc[1][1]
        st.session_state['act3_upper_y_new'] = df_upper.iloc[2][1]
        st.session_state['act4_upper_y_new'] = df_upper.iloc[3][1]
        st.session_state['act5_upper_y_new'] = df_upper.iloc[4][1]
        st.session_state['act6_upper_y_new'] = df_upper.iloc[5][1]
        
        st.session_state['act1_upper_z_new'] = df_upper.iloc[0][2]
        st.session_state['act2_upper_z_new'] = df_upper.iloc[1][2]
        st.session_state['act3_upper_z_new'] = df_upper.iloc[2][2]
        st.session_state['act4_upper_z_new'] = df_upper.iloc[3][2]
        st.session_state['act5_upper_z_new'] = df_upper.iloc[4][2]
        st.session_state['act6_upper_z_new'] = df_upper.iloc[5][2]
        
        st.session_state['act1_lower_x_new'] = df_lower.iloc[0][0]
        st.session_state['act2_lower_x_new'] = df_lower.iloc[1][0]
        st.session_state['act3_lower_x_new'] = df_lower.iloc[2][0]
        st.session_state['act4_lower_x_new'] = df_lower.iloc[3][0]
        st.session_state['act5_lower_x_new'] = df_lower.iloc[4][0]
        st.session_state['act6_lower_x_new'] = df_lower.iloc[5][0]
        
        st.session_state['act1_lower_y_new'] = df_lower.iloc[0][1]
        st.session_state['act2_lower_y_new'] = df_lower.iloc[1][1]
        st.session_state['act3_lower_y_new'] = df_lower.iloc[2][1]
        st.session_state['act4_lower_y_new'] = df_lower.iloc[3][1]
        st.session_state['act5_lower_y_new'] = df_lower.iloc[4][1]
        st.session_state['act6_lower_y_new'] = df_lower.iloc[5][1]
        
        st.session_state['act1_lower_z_new'] = df_lower.iloc[0][2]
        st.session_state['act2_lower_z_new'] = df_lower.iloc[1][2]
        st.session_state['act3_lower_z_new'] = df_lower.iloc[2][2]
        st.session_state['act4_lower_z_new'] = df_lower.iloc[3][2]
        st.session_state['act5_lower_z_new'] = df_lower.iloc[4][2]
        st.session_state['act6_lower_z_new'] = df_lower.iloc[5][2]
        
        st.session_state['minL_new'] = df_actuator.iloc[0][0]
        st.session_state['maxL_new'] = df_actuator.iloc[0][1]
        st.session_state['dist_frame_new'] = df_actuator.iloc[0][2]
        
if selected == 'Geometry':        
    
    st.header('Set the Geometry')

    if (st.button('Save changes', on_click=set_geometry, key='save_geo')):
        st.success("Changes are successfully saved!")

    if ('reset_geo_' in st.session_state):
        if st.session_state.reset_geo_ == False:
            if any([st.session_state[val+'_new'] != st.session_state[val] for val in dict_geo.keys()]):
                st.warning("You have something unsaved!")
                
    else:
        if any([st.session_state[val+'_new'] != st.session_state[val] for val in dict_geo.keys()]):
            st.warning("You have something unsaved!")
            
    with st.expander("Upper Frame (X-Y-Z)"):

        col1_upper, col2_upper, col3_upper = st.columns(3)    

        with col1_upper:
            
#             st.subheader('X')
            act1_upper_x = st.number_input(label='Actuator 1', value=st.session_state['act1_upper_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act1_upper_x')
            act2_upper_x = st.number_input(label='Actuator 2', value=st.session_state['act2_upper_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act2_upper_x')
            act3_upper_x = st.number_input(label='Actuator 3', value=st.session_state['act3_upper_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act3_upper_x')
            act4_upper_x = st.number_input(label='Actuator 4', value=st.session_state['act4_upper_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act4_upper_x')
            act5_upper_x = st.number_input(label='Actuator 5', value=st.session_state['act5_upper_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act5_upper_x')
            act6_upper_x = st.number_input(label='Actuator 6', value=st.session_state['act6_upper_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act6_upper_x')

        with col2_upper:

#             st.subheader('Y')
            act1_upper_y = st.number_input(label='', value=st.session_state['act1_upper_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act1_upper_y')
            act2_upper_y = st.number_input(label='', value=st.session_state['act2_upper_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act2_upper_y')
            act3_upper_y = st.number_input(label='', value=st.session_state['act3_upper_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act3_upper_y')
            act4_upper_y = st.number_input(label='', value=st.session_state['act4_upper_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act4_upper_y')
            act5_upper_y = st.number_input(label='', value=st.session_state['act5_upper_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act5_upper_y')
            act6_upper_y = st.number_input(label='', value=st.session_state['act6_upper_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act6_upper_y')

        with col3_upper:    

#             st.subheader('Z')
            act1_upper_z = st.number_input(label='', value=st.session_state['act1_upper_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act1_upper_z')
            act2_upper_z = st.number_input(label='', value=st.session_state['act2_upper_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act2_upper_z')
            act3_upper_z = st.number_input(label='', value=st.session_state['act3_upper_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act3_upper_z')
            act4_upper_z = st.number_input(label='', value=st.session_state['act4_upper_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act4_upper_z')
            act5_upper_z = st.number_input(label='', value=st.session_state['act5_upper_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act5_upper_z')
            act6_upper_z = st.number_input(label='', value=st.session_state['act6_upper_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act6_upper_z')

    with st.expander("Lower Frame (X-Y-Z)"):

        col1_lower, col2_lower, col3_lower = st.columns(3) 

        with col1_lower:

#             st.subheader('X') 
            act1_lower_x = st.number_input(label='Actuator 1', value=st.session_state['act1_lower_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act1_lower_x')
            act2_lower_x = st.number_input(label='Actuator 2', value=st.session_state['act2_lower_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act2_lower_x')
            act3_lower_x = st.number_input(label='Actuator 3', value=st.session_state['act3_lower_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act3_lower_x')
            act4_lower_x = st.number_input(label='Actuator 4', value=st.session_state['act4_lower_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act4_lower_x')
            act5_lower_x = st.number_input(label='Actuator 5', value=st.session_state['act5_lower_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act5_lower_x')
            act6_lower_x = st.number_input(label='Actuator 6', value=st.session_state['act6_lower_x_new']/st.session_state.unit, step=1.0, disabled=False, key='act6_lower_x')

        with col2_lower:

#             st.subheader('Y')
            act1_lower_y = st.number_input(label='', value=st.session_state['act1_lower_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act1_lower_y')
            act2_lower_y = st.number_input(label='', value=st.session_state['act2_lower_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act2_lower_y')
            act3_lower_y = st.number_input(label='', value=st.session_state['act3_lower_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act3_lower_y')
            act4_lower_y = st.number_input(label='', value=st.session_state['act4_lower_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act4_lower_y')
            act5_lower_y = st.number_input(label='', value=st.session_state['act5_lower_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act5_lower_y')
            act6_lower_y = st.number_input(label='', value=st.session_state['act6_lower_y_new']/st.session_state.unit, step=1.0, disabled=False, key='act6_lower_y')

        with col3_lower:

#             st.subheader('Z')
            act1_lower_z = st.number_input(label='', value=st.session_state['act1_lower_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act1_lower_z')
            act2_lower_z = st.number_input(label='', value=st.session_state['act2_lower_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act2_lower_z')
            act3_lower_z = st.number_input(label='', value=st.session_state['act3_lower_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act3_lower_z')
            act4_lower_z = st.number_input(label='', value=st.session_state['act4_lower_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act4_lower_z')
            act5_lower_z = st.number_input(label='', value=st.session_state['act5_lower_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act5_lower_z')
            act6_lower_z = st.number_input(label='', value=st.session_state['act6_lower_z_new']/st.session_state.unit, step=1.0, disabled=False, key='act6_lower_z')

    with st.expander("Other Parameters"):           

        col1_other, col2_other, col3_other = st.columns(3) 

        with col1_other:

#             st.subheader('Pivot Point')    
            pivot_x = st.number_input(label='Pivot Point - X', value=st.session_state['pivot_x_new']/st.session_state.unit, step=1.0, disabled=False, key='pivot_x')
            pivot_y = st.number_input(label='Pivot Point - Y', value=st.session_state['pivot_y_new']/st.session_state.unit, step=1.0, disabled=False, key='pivot_y')
            pivot_z = st.number_input(label='Pivot Point - Z', value=st.session_state['pivot_z_new']/st.session_state.unit, step=1.0, disabled=False, key='pivot_z')

        with col2_other:

#             st.subheader('Actuator Lengths')
            length_minimum = st.number_input(label='Minimum Actuator Length', value=st.session_state['minL_new']/st.session_state.unit, step=1.0, disabled=False, key='minL')
            length_maximum = st.number_input(label='Maximum Actuator Length', value=st.session_state['maxL_new']/st.session_state.unit, step=1.0, disabled=False, key='maxL')

        with col3_other:    

#             st.subheader('Frames')
            frame_distance = st.number_input(label='Distance Between Frames', value=st.session_state['dist_frame_new']/st.session_state.unit, step=1.0, disabled=False, key='dist_frame')

    if (st.button('Reset to default', on_click=reset_geo, key='reset_geo_')):
        st.success("Changes are successfully saved!")
#     try:
#         st.button('Reset to default', on_click=reset_geo, key='reset_geo_')
#     except:
#         pass
            
    
#     LIST1 = [[act1_upper_x, act2_upper_x, act3_upper_x, act4_upper_x, act5_upper_x, act6_upper_x],
#              [act1_upper_y, act2_upper_y, act3_upper_y, act4_upper_y, act5_upper_y, act6_upper_y],
#              [act1_upper_z, act2_upper_z, act3_upper_z, act4_upper_z, act5_upper_z, act6_upper_z],
#              [act1_lower_x, act2_lower_x, act3_lower_x, act4_lower_x, act5_lower_x, act6_lower_x],
#              [act1_lower_y, act2_lower_y, act3_lower_y, act4_lower_y, act5_lower_y, act6_lower_y],
#              [act1_lower_z, act2_lower_z, act3_lower_z, act4_lower_z, act5_lower_z, act6_lower_z]]
    
if selected == 'Results':

    st.header('Results')
    
    toggle = st.selectbox(label='Choose a plot option', options=("Actuator's initial length", "Admissible displacements ranges"), disabled=False)
    
    angles = np.array([[0.0], [0.0], [0.0]])
    linears = np.array([[0.0], [0.0], [0.0]])
    angles_0 = angles.copy()
    
    b_i = np.array([[st.session_state['act1_lower_x_new'], 
                     st.session_state['act2_lower_x_new'], 
                     st.session_state['act3_lower_x_new'], 
                     st.session_state['act4_lower_x_new'], 
                     st.session_state['act5_lower_x_new'], 
                     st.session_state['act6_lower_x_new']],
                    [st.session_state['act1_lower_y_new'], 
                     st.session_state['act2_lower_y_new'], 
                     st.session_state['act3_lower_y_new'], 
                     st.session_state['act4_lower_y_new'], 
                     st.session_state['act5_lower_y_new'], 
                     st.session_state['act6_lower_y_new']],
                    [st.session_state['act1_lower_z_new'], 
                     st.session_state['act2_lower_z_new'], 
                     st.session_state['act3_lower_z_new'], 
                     st.session_state['act4_lower_z_new'], 
                     st.session_state['act5_lower_z_new'], 
                     st.session_state['act6_lower_z_new']]])

    u_s = np.array([[st.session_state['act1_upper_x_new'], 
                     st.session_state['act2_upper_x_new'], 
                     st.session_state['act3_upper_x_new'], 
                     st.session_state['act4_upper_x_new'], 
                     st.session_state['act5_upper_x_new'], 
                     st.session_state['act6_upper_x_new']],
                    [st.session_state['act1_upper_y_new'], 
                     st.session_state['act2_upper_y_new'], 
                     st.session_state['act3_upper_y_new'], 
                     st.session_state['act4_upper_y_new'], 
                     st.session_state['act5_upper_y_new'], 
                     st.session_state['act6_upper_y_new']],
                    [st.session_state['act1_upper_z_new'], 
                     st.session_state['act2_upper_z_new'], 
                     st.session_state['act3_upper_z_new'], 
                     st.session_state['act4_upper_z_new'], 
                     st.session_state['act5_upper_z_new'], 
                     st.session_state['act6_upper_z_new']]])
    
    r_si_0 = np.array([[0], [0], [-st.session_state.dist_frame_new]])
    r_si = r_si_0 + linears
    
    initial = pd.DataFrame(np.array(actuator_length(angles_0, u_s, b_i, r_si)), columns=["Actuator's initial length (cm)"])
    initial = initial.round(decimals=1)
    initial = initial.rename(index={0: 'actuator 1', 1: 'actuator 2', 2: 'actuator 3', 3: 'actuator 4', 4: 'actuator 5', 5: 'actuator 6'})
    
    initial_ = pd.DataFrame(np.array(actuator_length(angles_0, u_s, b_i, r_si))/st.session_state.unit, columns=["Actuator's initial length (in)"])
    initial_ = initial_.round(decimals=1)
    initial_ = initial_.rename(index={0: 'actuator 1', 1: 'actuator 2', 2: 'actuator 3', 3: 'actuator 4', 4: 'actuator 5', 5: 'actuator 6'})

    if toggle == "Actuator's initial length":
        
        fig = plt.figure(figsize= (10,5)) 
        ax = fig.add_subplot(111)
        ax.bar(initial_.index, initial_["Actuator's initial length (in)"]*0+st.session_state.maxL_new/st.session_state.unit, color='lightgreen')
        pp = ax.bar(initial_.index, initial_["Actuator's initial length (in)"], color='lightcoral')
        ax.bar(initial_.index, initial_["Actuator's initial length (in)"]*0+st.session_state.minL_new/st.session_state.unit, color='white')
        ax.set_ylim([0.98*st.session_state.minL_new/st.session_state.unit,1.02*st.session_state.maxL_new/st.session_state.unit])
        for value in pp:
            height = value.get_height()
            ax.text(value.get_x() + value.get_width()/2., 1*height,'%d' % float((height-st.session_state.minL_new/st.session_state.unit)/(st.session_state.maxL_new/st.session_state.unit-st.session_state.minL_new/st.session_state.unit)*100) +'%', ha='center', va='bottom', fontsize=20)
        st.pyplot(fig)
            
#         if st.session_state.disp_unit_new == 'Inches':
#             fig = plt.figure(figsize= (10,5)) 
#             ax = fig.add_subplot(111)
#             ax.bar(initial_.index, initial_["Actuator's initial length (in)"]*0+st.session_state.maxL_new/st.session_state.unit, color='lightgreen')
#             pp = ax.bar(initial_.index, initial_["Actuator's initial length (in)"], color='lightcoral')
#             ax.bar(initial_.index, initial_["Actuator's initial length (in)"]*0+st.session_state.minL_new/st.session_state.unit, color='white')
#             ax.set_ylim([0.98*st.session_state.minL_new/st.session_state.unit,1.02*st.session_state.maxL_new/st.session_state.unit])
#             for value in pp:
#                 height = value.get_height()
#                 ax.text(value.get_x() + value.get_width()/2., 1*height,'%d' % float((height-st.session_state.minL_new/st.session_state.unit)/(st.session_state.maxL_new/st.session_state.unit-st.session_state.minL_new/st.session_state.unit)*100) +'%', ha='center', va='bottom', fontsize=20)
#             st.pyplot(fig)
#         else:
#             fig = plt.figure(figsize= (10,5)) 
#             ax = fig.add_subplot(111)
#             ax.bar(initial.index, initial["Actuator's initial length (cm)"]*0+st.session_state.maxL_new, color='lightgreen')
#             pp = ax.bar(initial.index, initial["Actuator's initial length (cm)"], color='lightcoral')
#             ax.bar(initial.index, initial["Actuator's initial length (cm)"]*0+st.session_state.minL_new, color='white')
#             ax.set_ylim([0.98*st.session_state.minL_new,1.02*st.session_state.maxL_new])
#             for value in pp:
#                 height = value.get_height()
#                 ax.text(value.get_x() + value.get_width()/2., 1*height,'%d' % float((height-st.session_state.minL_new)/(st.session_state.maxL_new-st.session_state.minL_new)*100) +'%', ha='center', va='bottom', fontsize=20)
#             st.pyplot(fig) 
            
        if st.session_state.disp_unit_new == 'Inches':  
            initial_ = initial_.style.set_properties(**{'text-align': 'center'})
            initial_ = initial_.set_table_styles([cell_hover, index_names, headers]) 
            initial_ = initial_.format({"Actuator's initial length (in)": "{}"})
          
            st.dataframe(initial_)
            
        else:
            initial = initial.style.set_properties(**{'text-align': 'center'})
            initial_new = initial.set_table_styles([cell_hover, index_names, headers])
            initial_new = initial_new.format({"Actuator's initial length (cm)": "{}"})
           
            st.dataframe(initial_new)
        
    elif toggle == "Admissible displacements ranges":

        ################
        
        value_final = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        dict_modes = {'surge':0, 'sway':1, 'heave':2, 'roll':3, 'pitch':4, 'yaw':5}
        dict_increment = {'min':0, 'max':1}
        act_len = np.zeros((6,6,2))

        PIV = np.array([[st.session_state.pivot_x], [st.session_state.pivot_y], [-st.session_state.pivot_z]])
        PIV_ext = np.array([[st.session_state.pivot_x], [st.session_state.pivot_y], [st.session_state.pivot_z]]).repeat(len(u_s[0]), axis=0).reshape(-1,len(u_s[0]))

        u_s_new = u_s - PIV_ext

        for j in dict_increment.keys():

            if j == 'min':
                increment = -0.1
            else:
                increment = +0.1

            for i in dict_modes.keys():

                flag = True
                value = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

                while flag == True:

                    linears = np.array([[value[dict_modes['surge']][dict_increment[j]]], [value[dict_modes['sway']][dict_increment[j]]], [-value[dict_modes['heave']][dict_increment[j]]]])
                    angles = np.array([[value[dict_modes['roll']][dict_increment[j]]], [value[dict_modes['pitch']][dict_increment[j]]], [value[dict_modes['yaw']][dict_increment[j]]]])
    #                 r_si = np.array([r_si_0[0], r_si_0[1], -r_si_0[2]]) + PIV + linears
                    r_si = r_si_0 + PIV + linears
                    actuator_lengths = np.array(actuator_length(angles, u_s_new, b_i, r_si))/st.session_state.unit

                    if (actuator_lengths.min() > st.session_state.minL_new/st.session_state.unit) & (actuator_lengths.max() < st.session_state.maxL_new/st.session_state.unit):
                        value[dict_modes[i]][dict_increment[j]] += increment
                    else:
                        flag = False   

                act_len[:,dict_modes[i],dict_increment[j]] = actuator_lengths
                value[dict_modes[i]][dict_increment[j]] -= increment

                value_final[dict_modes[i]][dict_increment[j]] = value[dict_modes[i]][dict_increment[j]]

        for i in range(6):
            if i < 3:
                value_final[i] = value_final[i]#*100
            else:
                value_final[i] = value_final[i]*180/np.pi

        value_final_new = value_final.copy()

        value_final_new = np.round(value_final_new, 1)/st.session_state.unit
        Admissible = pd.DataFrame(value_final_new, columns=['minimum range', 'maximum range'], index=['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw'])
        
        st.dataframe(Admissible)
        
        ################

def ResetMove():
    st.session_state.update({'surge_move': 0.0, 'sway_move': 0.0, 'heave_move': 0.0, 'roll_move': 0.0, 'pitch_move': 0.0, 'yaw_move': 0.0,})
            
if selected == 'Animation':

    st.header('Animation')
    
    angles = np.array([[st.session_state.roll_move], [st.session_state.pitch_move], [st.session_state.yaw_move]])
    linears = np.array([[st.session_state.surge_move], [st.session_state.sway_move], [st.session_state.heave_move]])
    angles_0 = angles.copy()
    
    b_i = np.array([[st.session_state['act1_lower_x_new'], 
                     st.session_state['act2_lower_x_new'], 
                     st.session_state['act3_lower_x_new'], 
                     st.session_state['act4_lower_x_new'], 
                     st.session_state['act5_lower_x_new'], 
                     st.session_state['act6_lower_x_new']],
                    [st.session_state['act1_lower_y_new'], 
                     st.session_state['act2_lower_y_new'], 
                     st.session_state['act3_lower_y_new'], 
                     st.session_state['act4_lower_y_new'], 
                     st.session_state['act5_lower_y_new'], 
                     st.session_state['act6_lower_y_new']],
                    [st.session_state['act1_lower_z_new'], 
                     st.session_state['act2_lower_z_new'], 
                     st.session_state['act3_lower_z_new'], 
                     st.session_state['act4_lower_z_new'], 
                     st.session_state['act5_lower_z_new'], 
                     st.session_state['act6_lower_z_new']]])

    u_s = np.array([[st.session_state['act1_upper_x_new'], 
                     st.session_state['act2_upper_x_new'], 
                     st.session_state['act3_upper_x_new'], 
                     st.session_state['act4_upper_x_new'], 
                     st.session_state['act5_upper_x_new'], 
                     st.session_state['act6_upper_x_new']],
                    [st.session_state['act1_upper_y_new'], 
                     st.session_state['act2_upper_y_new'], 
                     st.session_state['act3_upper_y_new'], 
                     st.session_state['act4_upper_y_new'], 
                     st.session_state['act5_upper_y_new'], 
                     st.session_state['act6_upper_y_new']],
                    [st.session_state['act1_upper_z_new'], 
                     st.session_state['act2_upper_z_new'], 
                     st.session_state['act3_upper_z_new'], 
                     st.session_state['act4_upper_z_new'], 
                     st.session_state['act5_upper_z_new'], 
                     st.session_state['act6_upper_z_new']]])
    
    r_si_0 = np.array([[0], [0], [-st.session_state.dist_frame_new]])
    r_si = r_si_0 + linears
    
    col1_animation, col2_animation = st.columns([1,3]) 

    with col1_animation:
        
        surge_move = st.number_input(label='Surge', value=0.0, step=0.5, disabled=False, key='surge_move')
        sway_move = st.number_input(label='Sway', value=0.0, step=0.5, disabled=False, key='sway_move')
        heave_move = st.number_input(label='Heave', value=0.0, step=0.5, disabled=False, key='heave_move')
        roll_move = st.number_input(label='Roll', value=0.0, step=0.05, disabled=False, key='roll_move')
        pitch_move = st.number_input(label='Pitch', value=0.0, step=0.05, disabled=False, key='pitch_move')
        yaw_move = st.number_input(label='Yaw', value=0.0, step=0.05, disabled=False, key='yaw_move')
        
        if (st.button('Reset', on_click=ResetMove, key='reset_move')):
            st.success("Reset!")

    with col2_animation:
        
        pivx = st.session_state.pivot_x
        pivy = st.session_state.pivot_y
        pivz = st.session_state.pivot_z
        
        PIV = np.array([[pivx], [pivy], [-pivz]])
        PIV_ext = np.array([[pivx], [pivy], [pivz]]).repeat(len(u_s[0]), axis=0).reshape(-1,len(u_s[0]))

        r_si_0_new = r_si_0 + PIV

        u_s_new = u_s - PIV_ext    

        r_si = linears
        u_s_updated = L_SI(u_s_new, angles) + r_si.repeat(len(u_s[0]), axis=0).reshape(-1,len(u_s[0]))
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')

        ax_, plot_ = plot_frame_new(u_s=u_s_new, b_i=b_i, r_si_0=r_si_0_new, ax=ax, scale=1, u_s_updated=u_s_updated, linewidths=[2,3])
                
        plot3d_x_lim_min = round(min(min(u_s[0,:])+r_si_0[0][0],min(b_i[0,:])),2)*1.5
        plot3d_x_lim_max = round(max(max(u_s[0,:])+r_si_0[0][0],max(b_i[0,:])),2)*1.5
        plot3d_y_lim_min = round(min(min(u_s[1,:])+r_si_0[1][0],min(b_i[1,:])),2)*1.5
        plot3d_y_lim_max = round(max(max(u_s[1,:])+r_si_0[1][0],max(b_i[1,:])),2)*1.5
        plot3d_z_lim_min = round(max(b_i[2,:]),2)*1.2
        plot3d_z_lim_max = round(-(min(u_s[2,:])+r_si_0[2][0]),2)*1.2
        
        plot_.xlim([plot3d_x_lim_min, plot3d_x_lim_max]) 
        plot_.ylim([plot3d_y_lim_min, plot3d_y_lim_max]) 
        ax_.set_zlim([plot3d_z_lim_min, plot3d_z_lim_max])
        ax_.scatter3D(st.session_state.pivot_x_new+st.session_state.surge_move, st.session_state.pivot_y_new+st.session_state.sway_move, st.session_state.dist_frame_new+st.session_state.pivot_z_new+st.session_state.heave_move, marker ='+', color='red', s=200)

        st.pyplot(fig)
        
        linears[2,0] = -linears[2,0]
        r_si = r_si_0 + linears
        
        initial = pd.DataFrame(np.array(actuator_length(angles_0, u_s, b_i, r_si)), columns=["Actuator's initial length (cm)"])
        initial = initial.round(decimals=1)
        initial = initial.rename(index={0: 'actuator 1', 1: 'actuator 2', 2: 'actuator 3', 3: 'actuator 4', 4: 'actuator 5', 5: 'actuator 6'})

        initial_ = pd.DataFrame(np.array(actuator_length(angles_0, u_s, b_i, r_si))/st.session_state.unit, columns=["Actuator's initial length (in)"])
        initial_ = initial_.round(decimals=1)
        initial_ = initial_.rename(index={0: 'actuator 1', 1: 'actuator 2', 2: 'actuator 3', 3: 'actuator 4', 4: 'actuator 5', 5: 'actuator 6'})

        if st.session_state.disp_unit_new == 'Inches':
            fig = plt.figure(figsize= (10,5)) 
            ax = fig.add_subplot(111)
            ax.bar(initial_.index, initial_["Actuator's initial length (in)"]*0+st.session_state.maxL_new/st.session_state.unit, color='lightgreen')
            pp = ax.bar(initial_.index, initial_["Actuator's initial length (in)"], color='lightcoral')
            ax.bar(initial_.index, initial_["Actuator's initial length (in)"]*0+st.session_state.minL_new/st.session_state.unit, color='white')
            ax.set_ylim([0.98*st.session_state.minL_new/st.session_state.unit,1.02*st.session_state.maxL_new/st.session_state.unit])
            for value in pp:
                height = value.get_height()
                ax.text(value.get_x() + value.get_width()/2., 1*height,'%d' % float((height-st.session_state.minL_new/st.session_state.unit)/(st.session_state.maxL_new/2.54-st.session_state.minL_new/st.session_state.unit)*100) +'%', ha='center', va='bottom', fontsize=20)
            st.pyplot(fig)
        else:
            fig = plt.figure(figsize= (8,4)) 
            ax = fig.add_subplot(111)
            ax.bar(initial.index, initial["Actuator's initial length (cm)"]*0+st.session_state.maxL_new, color='lightgreen')
            pp = ax.bar(initial.index, initial["Actuator's initial length (cm)"], color='lightcoral')
            ax.bar(initial.index, initial["Actuator's initial length (cm)"]*0+st.session_state.minL_new, color='white')
            ax.set_ylim([0.98*st.session_state.minL_new,1.02*st.session_state.maxL_new])
            for value in pp:
                height = value.get_height()
                ax.text(value.get_x() + value.get_width()/2., 1*height,'%d' % float((height-st.session_state.minL_new)/(st.session_state.maxL_new-st.session_state.minL_new)*100) +'%', ha='center', va='bottom', fontsize=20)
            st.pyplot(fig) 
        
def settings():
    st.session_state.update({'disp_unit_new': st.session_state.disp_unit})
    st.session_state.update({'rot_unit_new': st.session_state.rot_unit})
#     new_title = '<p style="font-family:sans-serif; color:Green; font-size: 14px; position: revert;">Changes are successfully saved!</p>'
#     st.write(new_title, unsafe_allow_html=True)

def reset_settings():
    st.session_state.update({'disp_unit_new': 'Centimeters'})
    st.session_state.update({'disp_unit': 'Centimeters'})
    st.session_state.update({'rot_unit_new': 'Degrees'})
    st.session_state.update({'rot_unit': 'Degrees'})

if selected == 'Settings':
    
    st.subheader('Units')

    if (st.button('Save changes', on_click=settings, key='save_geo')):
        st.success("Changes are successfully saved!")
    
    if ('reset_settings_' in st.session_state):
        if st.session_state.reset_settings_ == False:
            if any([st.session_state[val+'_new'] != st.session_state[val] for val in ['disp_unit', 'rot_unit']]):
                st.warning("You have something unsaved!")
    else:
        if any([st.session_state[val+'_new'] != st.session_state[val] for val in ['disp_unit', 'rot_unit']]):
            st.warning("You have something unsaved!")
    
    if st.session_state.disp_unit_new == 'Inches':
        disp_index = 1
    else:
        disp_index = 0
    st.radio(label="Displacements:", options=('Centimeters', 'Inches'), help='Selecte the rotations unit', index=disp_index, disabled=False, key='disp_unit')
    
    if st.session_state.rot_unit_new == 'Radians':
        rot_index = 1
    else:
        rot_index = 0
    st.radio(label="Rotations:", options=('Degrees', 'Radians'), help='Selecte the displacements unit', index=rot_index, disabled=False, key='rot_unit')
    
    if st.session_state.disp_unit_new == 'Inches':
        if st.session_state.disp_unit_flag == 'Centimeters':
            st.session_state.unit = 2.54
        st.session_state.disp_unit_flag = 'Inches'
    if st.session_state.disp_unit_new == 'Centimeters':
        if st.session_state.disp_unit_flag == 'Inches':
            st.session_state.unit = 1.0#/2.54
        st.session_state.disp_unit_flag = 'Centimeters'  
    
    if st.session_state.rot_unit_new == 'Radians':
        if st.session_state.rot_unit_flag == 'Degrees':
            st.session_state.unit_ = np.pi/180
        st.session_state.rot_unit_flag = 'Radians'
    if st.session_state.rot_unit_new == 'Degrees':
        if st.session_state.rot_unit_flag == 'Radians':
            st.session_state.unit_ = 1.0
        st.session_state.rot_unit_flag = 'Degrees'  
        
    if (st.button('Reset to default', on_click=reset_settings, key='reset_settings_')):
        st.success("Changes are successfully saved!")
        
from flask import Flask, request, render_template, redirect
import requests
import re
import numpy as np
from sklearn.externals import joblib
import folium

app = Flask(__name__)

app.vars={}

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        return redirect('/result')
    
@app.route('/result', methods=['GET','POST'])
def show_result():
        app.vars['address'] = request.form['address']
        app.vars['time'] = request.form['time']
        
        # Deal with missing field
        if not app.vars['address'] or not app.vars['time']:
            return render_template('error.html', message='Missing field')

        # Geocode address
        address = '+'.join(app.vars['address'].split())
        api_key = 'AIzaSyD6fHMPaIkXE8lIS0vMMUctX6n-44TK06I'
        base_url = 'https://maps.googleapis.com/maps/api/geocode/json?address='
        full_url = base_url + address + '&key='+ api_key
        r = requests.get(full_url, verify=False)
        geo = r.content
        
        if len(geo) == 52:
            return render_template('error.html', message='Invalid address')
        
        # Store geocodes
        latlon = re.findall(r'"location"\s+:\s+{\s+"lat"\s+:\s+(.*),\s+"lng"\s+:\s+(.*)\s+}', geo)
        app.vars['lat'] = float(latlon[0][0].strip("'"))
        app.vars['lon'] = float(latlon[0][1].strip("'"))
        
        # Read data
        dict_ID = {int(line.split(',')[0]): line.split(',')[1][:-1]
                   for line in open('models/pokemonID.csv','r').readlines()}
        pokeId = np.load('models/pokeId.npy')

        # Load model
        knn = joblib.load('models/model_kNN.pkl')
        # Predict
        prob = knn.predict_proba([app.vars['lat'], app.vars['lon']])
        sort_indices = np.argsort(prob, axis=1)[:,::-1][:,:10]
        app.vars['prob'] = prob[:,sort_indices[0]]
        app.vars['poke'] = [dict_ID[idx] for idx in pokeId[sort_indices[0]]]
        # Pack information into popup
        app.vars['text']='Address: %s\nLatitude: %f\nLongitude: %f\n'%(app.vars['address'],
                                                                app.vars['lat'],
                                                                app.vars['lon'])
        #for i in range(5):
        #    app.vars['text'] += '%s (%.3f), ' %(app.vars['poke'][i], app.vars['prob'][0][i])
        
        # Display and save prediction result on a map
        map_result = folium.Map(location=[app.vars['lat'], app.vars['lon']], 
                            zoom_start=11, tiles='Stamen Terrain')
        folium.Marker([app.vars['lat'], app.vars['lon']], popup=app.vars['text'],
                  icon = folium.Icon(icon = 'cloud', color ='red')).add_to(map_result)
        
        map_result.save('static/map_result.html')
        
        # save the data to file
        f = open('static/result.tsv', 'w')
        f.write('poke\tprob\n')
        for i in range(10):
            f.write("%s\t%f\n" %(app.vars['poke'][i], app.vars['prob'][0][i]))
        f.close()
        # Show result webpage
        return render_template('result.html')


@app.route('/tweet')
def show_tweet():
    return render_template('tweet.html')

@app.route('/about')
def show_about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    #app.run()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st
import time

st.header("Zararlı Bağlantı Tespiti")

url=st.text_input("Url girin (örnek: google.com)")
with st.spinner('bu işlem biraz zaman alabilir...'):
    time.sleep(5)
btn=st.button("Bağlantıyı kontrol et")

df=pd.read_csv("url_list.csv")
df=df[["url","durum"]]

urldf=pd.DataFrame({"url":url,"durum":0},index=[42])

df=pd.concat([df,urldf],ignore_index=True)

cv=CountVectorizer(max_features=50)
x=cv.fit_transform(df["url"]).toarray()
y=df["durum"]

tahmin=x[-1].copy()

x=x[0:-1]
y=y[0:-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,train_size=0.96)

randomf=RandomForestClassifier(n_estimators=800, max_depth=20, max_features='sqrt')
model=randomf.fit(x_train,y_train)

if btn:
    skor=model.score(x_test,y_test)
    sonuc=model.predict([tahmin])
    if 'guvenli' in sonuc:
        st.toast('kontrol ediliyor...')
        time.sleep(.5)
        st.toast('güvenli!', icon='✔️')
        msg="<p style='font-size:28px;color:#57ba6b'>Güvenli Bağlantı</p>"
        st.markdown(msg,unsafe_allow_html=True)
    else:
        st.toast('kontrol ediliyor...')
        time.sleep(.5)
        st.toast('zararlı!', icon='❌')
        msg = "<p style='font-size:28px;color:#ba6657'>Zararlı Bağlantı</p>"
        st.markdown(msg, unsafe_allow_html=True)
    st.text("Model skoru: ")
    st.text(skor)

footer="""<style>
a{
    text-decoration:none;
}
a:hover,a:active {
    background-color: white;
    border-radius:14px;
    text-decoration:none;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    height:55px;
    width: 100%;
    background-color:#141821;
    color: #fff;
    text-align: center;
}
</style>
<div class="footer">
<p style="margin-top:15px;"><span>Tuba Adıgüzel</span>
<a style='text-align: center;' href="https://github.com/tubaAdgzl" target="_blank"><svg xmlns="http://www.w3.org/2000/svg" height="20" width="19.375" viewBox="0 0 496 512"><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#731220" d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3 .3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5 .3-6.2 2.3zm44.2-1.7c-2.9 .7-4.9 2.6-4.6 4.9 .3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3 .7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3 .3 2.9 2.3 3.9 1.6 1 3.6 .7 4.3-.7 .7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3 .7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3 .7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>
</a></p>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)

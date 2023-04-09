{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "853b4bb9-0abc-4f19-86d8-5001d8987f21",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting streamlitNote: you may need to restart the kernel to use updated packages.\n",
      "  Downloading streamlit-1.21.0-py2.py3-none-any.whl (9.7 MB)\n",
      "Requirement already satisfied: tornado>=6.0.3 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (6.1)\n",
      "Requirement already satisfied: requests>=2.4 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (2.25.1)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "spyder 4.2.5 requires pyqt5<5.13, which is not installed.\n",
      "spyder 4.2.5 requires pyqtwebengine<5.13, which is not installed.\n",
      "arviz 0.11.4 requires typing-extensions<4,>=3.7.4.3, but you have typing-extensions 4.5.0 which is incompatible.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Requirement already satisfied: python-dateutil in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (2.8.1)\n",
      "Collecting cachetools>=4.0\n",
      "  Downloading cachetools-5.3.0-py3-none-any.whl (9.3 kB)\n",
      "Requirement already satisfied: numpy in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (1.20.1)\n",
      "Collecting blinker>=1.0.0\n",
      "  Downloading blinker-1.6.1-py3-none-any.whl (13 kB)\n",
      "Requirement already satisfied: click>=7.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (7.1.2)\n",
      "Collecting validators>=0.2\n",
      "  Downloading validators-0.20.0.tar.gz (30 kB)\n",
      "Collecting protobuf<4,>=3.12\n",
      "  Downloading protobuf-3.20.3-cp38-cp38-win_amd64.whl (904 kB)\n",
      "Collecting gitpython!=3.1.19\n",
      "  Downloading GitPython-3.1.31-py3-none-any.whl (184 kB)\n",
      "Collecting pyarrow>=4.0\n",
      "  Downloading pyarrow-11.0.0-cp38-cp38-win_amd64.whl (20.6 MB)\n",
      "Requirement already satisfied: packaging>=14.1 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (20.9)\n",
      "Requirement already satisfied: toml in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Collecting tzlocal>=1.1\n",
      "  Downloading tzlocal-4.3-py3-none-any.whl (20 kB)\n",
      "Collecting pympler>=0.9\n",
      "  Downloading Pympler-1.0.1-py3-none-any.whl (164 kB)\n",
      "Collecting rich>=10.11.0\n",
      "  Downloading rich-13.3.3-py3-none-any.whl (238 kB)\n",
      "Requirement already satisfied: importlib-metadata>=1.4 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (3.10.0)\n",
      "Requirement already satisfied: altair<5,>=3.2.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (4.2.0)\n",
      "Requirement already satisfied: pandas<2,>=0.25 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (1.2.4)\n",
      "Requirement already satisfied: pillow>=6.2.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (8.2.0)\n",
      "Requirement already satisfied: watchdog in c:\\users\\miche\\anaconda3\\lib\\site-packages (from streamlit) (1.0.2)\n",
      "Collecting typing-extensions>=3.10.0.0\n",
      "  Downloading typing_extensions-4.5.0-py3-none-any.whl (27 kB)\n",
      "Collecting pydeck>=0.1.dev5\n",
      "  Downloading pydeck-0.8.0-py2.py3-none-any.whl (4.7 MB)\n",
      "Requirement already satisfied: entrypoints in c:\\users\\miche\\anaconda3\\lib\\site-packages (from altair<5,>=3.2.0->streamlit) (0.3)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from altair<5,>=3.2.0->streamlit) (3.2.0)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from altair<5,>=3.2.0->streamlit) (2.11.3)\n",
      "Requirement already satisfied: toolz in c:\\users\\miche\\anaconda3\\lib\\site-packages (from altair<5,>=3.2.0->streamlit) (0.11.1)\n",
      "Collecting gitdb<5,>=4.0.1\n",
      "  Downloading gitdb-4.0.10-py3-none-any.whl (62 kB)\n",
      "Collecting smmap<6,>=3.0.1\n",
      "  Downloading smmap-5.0.0-py3-none-any.whl (24 kB)\n",
      "Requirement already satisfied: zipp>=0.5 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from importlib-metadata>=1.4->streamlit) (3.4.1)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit) (20.3.0)\n",
      "Requirement already satisfied: six>=1.11.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit) (1.15.0)\n",
      "Requirement already satisfied: setuptools in c:\\users\\miche\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit) (52.0.0.post20210125)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit) (0.17.3)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from packaging>=14.1->streamlit) (2.4.7)\n",
      "Requirement already satisfied: pytz>=2017.3 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from pandas<2,>=0.25->streamlit) (2021.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from jinja2->altair<5,>=3.2.0->streamlit) (1.1.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from requests>=2.4->streamlit) (2020.12.5)\n",
      "Requirement already satisfied: idna<3,>=2.5 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from requests>=2.4->streamlit) (2.10)\n",
      "Requirement already satisfied: chardet<5,>=3.0.2 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from requests>=2.4->streamlit) (4.0.0)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from requests>=2.4->streamlit) (1.26.4)\n",
      "Collecting pygments<3.0.0,>=2.13.0\n",
      "  Downloading Pygments-2.14.0-py3-none-any.whl (1.1 MB)\n",
      "Collecting markdown-it-py<3.0.0,>=2.2.0\n",
      "  Downloading markdown_it_py-2.2.0-py3-none-any.whl (84 kB)\n",
      "Collecting mdurl~=0.1\n",
      "  Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)\n",
      "Collecting tzdata\n",
      "  Downloading tzdata-2023.3-py2.py3-none-any.whl (341 kB)\n",
      "Collecting backports.zoneinfo\n",
      "  Downloading backports.zoneinfo-0.2.1-cp38-cp38-win_amd64.whl (38 kB)\n",
      "Collecting pytz-deprecation-shim\n",
      "  Downloading pytz_deprecation_shim-0.1.0.post0-py2.py3-none-any.whl (15 kB)\n",
      "Requirement already satisfied: decorator>=3.4.0 in c:\\users\\miche\\anaconda3\\lib\\site-packages (from validators>=0.2->streamlit) (5.0.6)\n",
      "Building wheels for collected packages: validators\n",
      "  Building wheel for validators (setup.py): started\n",
      "  Building wheel for validators (setup.py): finished with status 'done'\n",
      "  Created wheel for validators: filename=validators-0.20.0-py3-none-any.whl size=19567 sha256=5ea24220dc55547255bb28535b983bc5d21de9dc812bb39d577ade8590b89098\n",
      "  Stored in directory: c:\\users\\miche\\appdata\\local\\pip\\cache\\wheels\\19\\09\\72\\3eb74d236bb48bd0f3c6c3c83e4e0c5bbfcbcad7c6c3539db8\n",
      "Successfully built validators\n",
      "Installing collected packages: tzdata, smmap, mdurl, backports.zoneinfo, typing-extensions, pytz-deprecation-shim, pygments, markdown-it-py, gitdb, validators, tzlocal, rich, pympler, pydeck, pyarrow, protobuf, gitpython, cachetools, blinker, streamlit\n",
      "  Attempting uninstall: typing-extensions\n",
      "    Found existing installation: typing-extensions 3.7.4.3\n",
      "    Uninstalling typing-extensions-3.7.4.3:\n",
      "      Successfully uninstalled typing-extensions-3.7.4.3\n",
      "  Attempting uninstall: pygments\n",
      "    Found existing installation: Pygments 2.8.1\n",
      "    Uninstalling Pygments-2.8.1:\n",
      "      Successfully uninstalled Pygments-2.8.1\n",
      "Successfully installed backports.zoneinfo-0.2.1 blinker-1.6.1 cachetools-5.3.0 gitdb-4.0.10 gitpython-3.1.31 markdown-it-py-2.2.0 mdurl-0.1.2 protobuf-3.20.3 pyarrow-11.0.0 pydeck-0.8.0 pygments-2.14.0 pympler-1.0.1 pytz-deprecation-shim-0.1.0.post0 rich-13.3.3 smmap-5.0.0 streamlit-1.21.0 typing-extensions-4.5.0 tzdata-2023.3 tzlocal-4.3 validators-0.20.0\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3087d061-79d6-406e-b1d7-8ebe36e82a62",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import nltk\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "\n",
    "# Load the trained model and vectorizer\n",
    "stemmer = PorterStemmer()\n",
    "tfidf = pickle.load(open('vectorizer.pkl', 'rb'))\n",
    "model = pickle.load(open('model.pkl', 'rb'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ab0d5875-e2ff-4a44-9f65-92c871386ea1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define a function to preprocess the input text\n",
    "def preprocess_text(text):\n",
    "    import string\n",
    "\n",
    "    # Convert text to lowercase and tokenize into words\n",
    "    words = nltk.word_tokenize(text.lower())\n",
    "\n",
    "    # Remove stopwords and punctuation\n",
    "    stopwords_set = set(nltk.corpus.stopwords.words('english'))\n",
    "    words = [\n",
    "        w for w in words if w not in stopwords_set and w not in string.punctuation]\n",
    "\n",
    "    # Stem words using PorterStemmer\n",
    "    words = [stemmer.stem(w) for w in words]\n",
    "\n",
    "    # Join the preprocessed words into a single string\n",
    "    return \" \".join(words)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "43f97ad0-85d1-420d-baf4-88e41bf86f6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define a function to make predictions\n",
    "def predict_spam(input_text):\n",
    "    # Preprocess the input text\n",
    "    preprocessed_text = preprocess_text(input_text)\n",
    "\n",
    "    # Vectorize the preprocessed text\n",
    "    vector_input = tfidf.transform([preprocessed_text])\n",
    "\n",
    "    # Make the prediction\n",
    "    prediction = model.predict(vector_input)[0]\n",
    "\n",
    "    # Get the prediction probability\n",
    "    prediction_prob = model.predict_proba(vector_input)[0][1]\n",
    "\n",
    "    return prediction, prediction_prob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "97af5caa-30f3-4a1e-99e9-483234ec9c8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the Streamlit app\n",
    "def app():\n",
    "    # Set the app title and favicon\n",
    "    st.set_page_config(page_title='Spam Classifier', page_icon=':envelope:')\n",
    "\n",
    "    # Add a header with the app title\n",
    "    st.write('# Spam Classifier')\n",
    "    st.write('---')\n",
    "\n",
    "    # Add a logo\n",
    "    st.image('logo.png', width=200)\n",
    "\n",
    "    # Add a text input field for the user to enter their message\n",
    "    input_sms = st.text_input('Enter your message here:')\n",
    "\n",
    "    # Add input validation\n",
    "    if not input_sms:\n",
    "        st.warning('Please enter a message.')\n",
    "        st.stop()\n",
    "\n",
    "    # Add instructions\n",
    "    st.write('Please enter a message to predict if it is spam or not.')\n",
    "    st.write('---')\n",
    "\n",
    "    # Add a predict button\n",
    "    if st.button('Predict'):\n",
    "        # Add a loading spinner while the prediction is being made\n",
    "        with st.spinner('Predicting...'):\n",
    "            # Make the prediction\n",
    "            prediction, prediction_prob = predict_spam(input_sms)\n",
    "\n",
    "        # Display the prediction result\n",
    "        if prediction == 1:\n",
    "            st.error(\n",
    "                f'This message is spam with a probability of {prediction_prob:.2f}.')\n",
    "        else:\n",
    "            st.success(\n",
    "                f'This message is not spam with a probability of {1 - prediction_prob:.2f}.')\n",
    "\n",
    "    # Add a reset button to clear the input field\n",
    "    if st.button('Reset'):\n",
    "        input_sms = ''\n",
    "        st.warning('Input field cleared.')\n",
    "\n",
    "    # Add a horizontal line to separate the sections\n",
    "    st.write('---')\n",
    "\n",
    "    # Add some styling to the footer\n",
    "    st.markdown('<p style=\"font-size:10px; font-style:italic; text-align:center;\">Built with Streamlit by Your Name</p>', unsafe_allow_html=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a0a0525a-14f2-4ae8-bd50-0a63cb1b88e2",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "StopException",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mStopException\u001b[0m                             Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-8-fb84fa1e9c1c>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'__main__'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m     \u001b[0mapp\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-7-cff795b3cc5b>\u001b[0m in \u001b[0;36mapp\u001b[1;34m()\u001b[0m\n\u001b[0;32m     17\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0minput_sms\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     18\u001b[0m         \u001b[0mst\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mwarning\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Please enter a message.'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 19\u001b[1;33m         \u001b[0mst\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     20\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     21\u001b[0m     \u001b[1;31m# Add instructions\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\commands\\execution_control.py\u001b[0m in \u001b[0;36mstop\u001b[1;34m()\u001b[0m\n\u001b[0;32m     41\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     42\u001b[0m     \"\"\"\n\u001b[1;32m---> 43\u001b[1;33m     \u001b[1;32mraise\u001b[0m \u001b[0mStopException\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     44\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     45\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mStopException\u001b[0m: "
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    app()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cff6e2ac-5b4d-4a58-ba21-23e04f0e27bd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

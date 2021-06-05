import streamlit as st
from fastai.vision.all import *

st.set_page_config(
    page_title="Prediction App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

def is_cat(x): return x[0].isupper()
learn = load_learner('model.pkl')


def main():
    st.title('Upload an image')
    image_file = st.file_uploader('Upload a image file', ['png','jpg','jpeg'], accept_multiple_files=False)
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        # st.write(file_details)
        image = load_image(image_file)

        img = PILImage.create(image_file)
        is_cat, _, probs = learn.predict(img)
        print(f"Is this a cat?: {is_cat}.")
        print(f"Probability it's a cat: {probs[1].item():.6f}")
        st.write(f"Is this a cat?: {is_cat}.")
        st.write(f"Probability it's a cat: {probs[1].item():.6f}")
        


        st.image(image)
    
if __name__ == main():
    main()










# def helper():
#     img = PILImage.create(filename)
#     image = Image.open(filename)
#     is_cat, _, probs = learn.predict(img)
#     # print(f"Is this a cat?: {is_cat}.")
#     # print(f"Probability it's a cat: {probs[1].item():.6f}")
#     st.write(f"Is this a cat?: {is_cat}.")
#     st.write(f"Probability it's a cat: {probs[1].item():.6f}")
#     st.image(image, caption='Cat or not',use_column_width=True)


# if(filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png')):
#     helper()




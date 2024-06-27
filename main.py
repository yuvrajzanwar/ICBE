import uvicorn
from fastapi import FastAPI, File, UploadFile,Form
import warnings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from script import generate_caption
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


warnings.filterwarnings("ignore")

@app.get("/")
def home():
    return {"hello": "HELLO WORLD"}

@app.post("/get_caption")
def get_caption(file: bytes = File(...), prompt: str = Form(...)):
    print("GOT IMAGE")
    # print("\n\n",file)
    contents = file
    # prompt=file['prompt']
    print(prompt)
    pil_image = Image.open(io.BytesIO(contents)).resize((224, 224))
    image_array = img_to_array(pil_image)
    try :
        cap, captions = generate_caption(image_array,prompt)
    except Exception as e:
        return {
            "error": f"Error processing image: {str(e)}"
        }
    # print("\n")
    print("Image Caption: ", cap)
    print("Suggested Captions: ", captions)
    
    return {
        "caption": cap,
        "suggestions": captions,
    }
    

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000, debug=True)

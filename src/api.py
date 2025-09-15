import sqlite3
import datetime
import base64
from io import BytesIO

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

# Create the APIRouter instance. All API routes will be attached to this.
router = APIRouter()


def predict_anomaly(image: Image.Image) -> str:
    """
    Placeholder function for the Machine Learning model.
    
    In a real application, you would load your trained model (e.g., TensorFlow,
    PyTorch) and perform inference on the preprocessed image.

    Args:
        image (PIL.Image.Image): The input image from the user upload.

    Returns:
        str: The prediction result, either "Normal" or "Anomaly Detected".
    """
    # For demonstration, we'll simulate a prediction.
    # Replace this with your actual model prediction logic.
    if datetime.datetime.now().second % 2 == 0:
        return "Anomaly Detected"
    else:
        return "Normal"


@router.post("/submit", summary="Submit a new inspection")
async def submit_inspection(
    qr_code: str = Form(...),
    inspector_name: str = Form(...),
    gps_location: str = Form(...),
    inspection_photo: UploadFile = File(...)
):
    """
    Receives inspection data and an image, gets a prediction from the ML
    model, and saves everything to the database.
    """
    try:
        # 1. Read and process the uploaded image
        contents = await inspection_photo.read()
        image = Image.open(BytesIO(contents))
        
        # 2. Get a prediction from the ML model
        prediction = predict_anomaly(image)

        # 3. Convert image to a base64 string to store in the database
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 4. Connect to the database and save the record
        conn = sqlite3.connect("inspection.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO inspections (qr_code, inspector_name, timestamp, gps_location, inspection_photo, ml_prediction)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                qr_code,
                inspector_name,
                datetime.datetime.now(),
                gps_location,
                img_base64_str,
                prediction,
            ),
        )
        conn.commit()
        conn.close()

        return JSONResponse(
            status_code=200,
            content={"message": "Inspection submitted successfully", "prediction": prediction}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )


@router.get("/dashboard", response_class=HTMLResponse, summary="View all inspections")
def dashboard():
    """
    Fetches all inspection records from the database and displays them
    in an HTML table.
    """
    conn = sqlite3.connect("inspection.db")
    # Use Row factory to access columns by name (e.g., row['qr_code'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inspections ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    # --- Build the HTML response ---
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Inspection Dashboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; }
            .container { max-width: 1200px; margin: auto; }
            h2 { color: #343a40; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background-color: white; }
            th, td { border: 1px solid #dee2e6; padding: 12px; text-align: center; vertical-align: middle; }
            th { background-color: #e9ecef; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #e2e6ea; }
            .anomaly { color: #dc3545; font-weight: bold; }
            .normal { color: #28a745; font-weight: bold; }
            img { max-width: 120px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Inspection Dashboard</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>QR Code</th>
                        <th>Inspector</th>
                        <th>Timestamp</th>
                        <th>GPS</th>
                        <th>Photo</th>
                        <th>Prediction</th>
                    </tr>
                </thead>
                <tbody>
    """

    for row in rows:
        img_tag = f'<img src="data:image/png;base64,{row["inspection_photo"]}" alt="Inspection Photo" />' if row["inspection_photo"] else "No Photo"
        
        pred_class = "anomaly" if row["ml_prediction"] == "Anomaly Detected" else "normal"
        pred_icon = "⚠️" if pred_class == "anomaly" else "✅"
        pred_text = row["ml_prediction"].replace(" Detected", "")
        pred_tag = f'<span class="{pred_class}">{pred_icon} {pred_text}</span>'
        
        html += f"""
            <tr>
                <td>{row["id"]}</td>
                <td>{row["qr_code"]}</td>
                <td>{row["inspector_name"]}</td>
                <td>{row["timestamp"]}</td>
                <td>{row["gps_location"]}</td>
                <td>{img_tag}</td>
                <td>{pred_tag}</td>
            </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)
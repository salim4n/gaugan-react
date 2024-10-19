import { useEffect, useRef, useState } from "react"
import * as tf from "@tensorflow/tfjs"
import { ChromePicker } from "react-color"

const gauganUrl =
  "https://huggingface.co/salim4n/gaugan-tfjs/resolve/main/model.json"

// Définir des tailles constantes pour les canvas
const CANVAS_WIDTH = 256
const CANVAS_HEIGHT = 256

const GAUGAN_TAGS = [
  "sky",
  "cloud",
  "grass",
  "tree",
  "mountain",
  "water",
  "earth",
  "road",
  "rock",
  "sand",
  "snow",
  "building",
  "bush",
  "flower",
  "sea",
  "river",
  "hill",
  "forest",
] as const

function App() {
  const [gaugan, setGaugan] =
    useState<tf.GraphModel<string | tf.io.IOHandler>>()
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [color, setColor] = useState("#ffffff")
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [preHeating, setPreHeating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)

  const startDrawing = (event: React.MouseEvent) => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!ctx) return

    setIsDrawing(true)
    ctx.strokeStyle = color
    ctx.lineWidth = 5
    ctx.lineJoin = "round"
    ctx.lineCap = "round"
    ctx.beginPath()
    // Ajuster les coordonnées en fonction de la taille du canvas
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) * (canvas.width / rect.width)
    const y = (event.clientY - rect.top) * (canvas.height / rect.height)
    ctx.moveTo(x, y)
  }

  const stopDrawing = () => {
    setIsDrawing(false)
  }

  const draw = (event: React.MouseEvent) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!ctx) return

    // Ajuster les coordonnées en fonction de la taille du canvas
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) * (canvas.width / rect.width)
    const y = (event.clientY - rect.top) * (canvas.height / rect.height)
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  const encodeGauGANTags = (): tf.Tensor2D => {
    const encoding = new Array(192).fill(0)
    GAUGAN_TAGS.forEach((_, index) => {
      encoding[index] = 1
    })
    return tf.tensor2d([encoding], [1, 192])
  }

  const generateImage = async () => {
    if (!canvasRef.current || !gaugan) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    let inputTensor = tf.browser
      .fromPixels(imageData)
      .toFloat()
      .div(tf.scalar(255))

    // Le redimensionnement n'est plus nécessaire car le canvas est déjà à 256x256
    inputTensor = inputTensor.expandDims(0)

    const tagsTensor = encodeGauGANTags()
    const tensor1 = tf.tidy(() => tagsTensor)
    const tensor2 = tf.tidy(() => {
      const imageChannels = inputTensor.slice([0, 0, 0, 0], [1, 256, 256, 3])
      const remainingChannels = tf.zeros([1, 256, 256, 15])
      return tf.concat([imageChannels, remainingChannels], 3)
    })

    const output = (await gaugan.predict([tensor1, tensor2])) as tf.Tensor
    tf.dispose([tensor1, tensor2, inputTensor, tagsTensor])

    const imageTensor = output
      .squeeze()
      .mul(tf.scalar(255))
      .clipByValue(0, 255)
      .toInt()
    const imageDataOutput = await tf.browser.toPixels(
      imageTensor as tf.Tensor3D
    )

    const newCanvas = document.createElement("canvas")
    newCanvas.width = CANVAS_WIDTH
    newCanvas.height = CANVAS_HEIGHT
    const newCtx = newCanvas.getContext("2d")
    if (newCtx) {
      const newImageData = newCtx.createImageData(CANVAS_WIDTH, CANVAS_HEIGHT)
      newImageData.data.set(imageDataOutput)
      newCtx.putImageData(newImageData, 0, 0)
      const generatedURL = newCanvas.toDataURL()
      setGeneratedImage(generatedURL)
    }

    tf.dispose([output, imageTensor])
  }

  useEffect(() => {
    ;(async () => {
      tf.setBackend("webgl").then(() => console.log(tf.getBackend()))
      setLoading(true)
      const indexedDB = await window.indexedDB.databases()

      if (indexedDB.some((db: IDBDatabaseInfo) => db.name === "tensorflowjs")) {
        const model = await tf.loadGraphModel("indexeddb://gaugan-tfjs", {
          onProgress: progress => setProgress(progress),
        })
        setPreHeating(true)
        const tensor1 = tf.zeros([1, 192])
        const tensor2 = tf.zeros([1, 256, 256, 18])
        const pred = model.predict([tensor1, tensor2])
        setPreHeating(false)
        tf.dispose([tensor1, tensor2, pred])
        setGaugan(model)
        setLoading(false)
      } else {
        const model = await tf.loadGraphModel(gauganUrl, {
          onProgress: progress => setProgress(progress),
        })
        await model.save("indexeddb://gaugan-tfjs")
        setGaugan(model)
        setLoading(false)
      }
    })()
  }, [])

  if (preHeating) {
    return (
      <h1 style={{ color: "white", textAlign: "center" }}>
        🌡️ pre-heating model...
      </h1>
    )
  }

  return (
    <div
      className="app-container"
      style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ color: "white", textAlign: "center" }}>
        GauGAN TensorFlow.js
      </h1>

      {loading && (
        <div style={{ textAlign: "center", marginTop: "20px" }}>
          <span
            style={{ color: "#e74c3c", fontSize: "20px", marginRight: "10px" }}>
            {Math.round(progress * 100)}%
          </span>
          <progress
            value={progress * 100}
            max="100"
            style={{ width: "100%" }}></progress>
          <p style={{ color: "#888" }}>Loading model...</p>
        </div>
      )}

      <div
        style={{
          marginTop: "30px",
          textAlign: "center",
          padding: "20px",
          display: "flex",
          justifyContent: "center",
        }}>
        <div
          style={{
            display: "inline-block",
            padding: "10px",
            backgroundColor: "#fff",
            cursor: "crosshair",
          }}>
          <ChromePicker
            color={color}
            onChangeComplete={(newColor: { hex: string }) =>
              setColor(newColor.hex)
            }
          />
        </div>
      </div>

      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <button
          onClick={generateImage}
          style={{
            backgroundColor: "#e74c3c",
            color: "white",
            padding: "10px 20px",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
            fontSize: "16px",
          }}>
          Generate Image
        </button>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          marginTop: "30px",
        }}>
        <div style={{ textAlign: "center", marginTop: "30px" }}>
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            style={{
              cursor: "crosshair",
              backgroundColor: "white",
              borderRadius: "10px",
              width: "512px",
              height: "512px",
              imageRendering: "pixelated", // Garde les pixels nets lors du redimensionnement
            }}
          />
        </div>

        {generatedImage && (
          <div style={{ marginTop: "30px", textAlign: "center" }}>
            <h3 style={{ color: "white" }}>Generated Image:</h3>
            <img
              src={generatedImage}
              alt="Generated"
              style={{
                width: "512px", // Double size for display
                height: "512px", // Double size for display
                borderRadius: "10px",
                boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
                imageRendering: "pixelated", // Garde les pixels nets lors du redimensionnement
              }}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default App

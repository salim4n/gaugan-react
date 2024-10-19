import * as tf from "@tensorflow/tfjs"
import { ChromePicker } from "react-color"
import { GAUGAN_TAGS, CANVAS_HEIGHT, CANVAS_WIDTH } from "./sd"
import useHook from "./hook"
import { CSSProperties, useRef, useState } from "react"

type GauGANTag = (typeof GAUGAN_TAGS)[number]

function App() {
  const { gaugan, progress, loading, preHeating } = useHook()
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [color, setColor] = useState<string>("#ffffff")
  const [isDrawing, setIsDrawing] = useState<boolean>(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [urlCss, setUrlCss] = useState<CSSProperties>({
    color: "lightblue",
    font: "small-caption",
    fontSize: "20px",
  })
  const [heartCss, setHeartCss] = useState<CSSProperties>({})
  const [selectedTags, setSelectedTags] = useState<GauGANTag[]>([])

  const startDrawing = (event: React.MouseEvent): void => {
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

  const stopDrawing = (): void => setIsDrawing(false)

  const draw = (event: React.MouseEvent): void => {
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

  const toggleTag = (tag: GauGANTag) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    )
  }

  const encodeGauGANTags = (): tf.Tensor2D => {
    const encoding = new Array(192).fill(0)
    selectedTags.forEach(tag => {
      const index = GAUGAN_TAGS.indexOf(tag)
      if (index !== -1) {
        encoding[index] = 1
      }
    })
    return tf.tensor2d([encoding], [1, 192])
  }

  const generateImage = async (): Promise<void> => {
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
        Gaugan - React - Tensorflow.js
      </h1>
      <p style={{ textAlign: "center", color: "white" }}>
        A generative art tool for creating unique and beautiful images.
      </p>
      <p
        style={{
          textAlign: "center",
          font: "small-caps",
          color: "white",
        }}>
        Made with{" "}
        <span
          style={heartCss}
          onMouseEnter={() => {
            setHeartCss({
              fontSize: "30px",
              transform: "rotate(1.5turn)",
              transition: "all 0.2s ease-in-out",
            })
          }}
          onMouseLeave={() => {
            setHeartCss({
              fontSize: "20px",
              transform: "rotate(1turn)",
              transition: "all 3s ease-in-out",
            })
          }}>
          ❤️
        </span>{" "}
        by{" "}
        <a
          onMouseEnter={() => {
            setUrlCss({
              color: "gold",
              font: "icon",
              fontSize: "30px",
              transition: "all 0.2s ease-in-out",
            })
            setHeartCss({
              fontSize: "30px",
              transform: "scale(1.5)",
              transition: "all 0.2s ease-in-out",
            })
          }}
          onMouseLeave={() => {
            setUrlCss({
              color: "lightblue",
              fontSize: "20px",
              font: "small-caption",
              transition: "all 0.2s ease-in-out",
            })
            setHeartCss({
              fontSize: "20px",
              transform: "scale(1)",
              transition: "all 3s ease-in-out",
            })
          }}
          href="https://github.com/salim4n"
          target="_blank"
          style={urlCss}>
          Salim Laimeche
        </a>
      </p>

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
      {/* Tag Selection */}
      <div
        style={{
          marginTop: "20px",
          textAlign: "center",
          display: "flex",
          flexWrap: "wrap",
          gap: "10px",
          justifyContent: "center",
          padding: "20px",
        }}>
        {GAUGAN_TAGS.map(tag => (
          <button
            key={tag}
            onClick={() => toggleTag(tag)}
            style={{
              padding: "8px 16px",
              borderRadius: "20px",
              border: "none",
              backgroundColor: selectedTags.includes(tag) ? "#e74c3c" : "#555",
              color: "white",
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}>
            {tag}
          </button>
        ))}
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
          margin: "20px auto",
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
              imageRendering: "pixelated",
            }}
          />
        </div>

        {generatedImage && (
          <div
            style={{
              marginTop: "30px",
              textAlign: "center",
              marginLeft: "50px",
            }}>
            <img
              src={generatedImage}
              alt="Generated"
              style={{
                width: "512px", // Double size for display
                height: "512px", // Double size for display
                borderRadius: "10px",
                imageRendering: "pixelated",
              }}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default App

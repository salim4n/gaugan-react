import * as tf from "@tensorflow/tfjs"
import useHook from "./hook"
import { CSSProperties, useRef, useState } from "react"
import { CANVAS_HEIGHT, CANVAS_WIDTH, GauGANTag, TAG_COLORS } from "./sd"

function App() {
  const { gaugan, progress, loading, preHeating } = useHook()
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [isDrawing, setIsDrawing] = useState<boolean>(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [urlCss, setUrlCss] = useState<CSSProperties>({
    color: "lightblue",
    font: "small-caption",
    fontSize: "20px",
  })
  const [heartCss, setHeartCss] = useState<CSSProperties>({})
  const [currentColor, setCurrentColor] = useState<string>(
    TAG_COLORS.grass.color
  )
  const [selectedBackground, setSelectedBackground] = useState<GauGANTag>("sky")
  const [currentTag, setCurrentTag] = useState<GauGANTag>("grass")
  const [activePixels, setActivePixels] = useState<Set<GauGANTag>>(new Set())

  const startDrawing = (event: React.MouseEvent): void => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!ctx) return

    setIsDrawing(true)
    ctx.strokeStyle = currentColor
    ctx.lineWidth = 5
    ctx.lineJoin = "round"
    ctx.lineCap = "round"
    ctx.beginPath()
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) * (canvas.width / rect.width)
    const y = (event.clientY - rect.top) * (canvas.height / rect.height)
    ctx.moveTo(x, y)
    setActivePixels(prev => new Set([...prev, currentTag]))
  }

  const stopDrawing = (): void => setIsDrawing(false)

  const draw = (event: React.MouseEvent) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!ctx) return

    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) * (canvas.width / rect.width)
    const y = (event.clientY - rect.top) * (canvas.height / rect.height)
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!ctx) return
    setSelectedBackground("sky")
    ctx.fillStyle = TAG_COLORS[selectedBackground].color
    if (!canvas) return
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    setActivePixels(new Set())
  }

  const selectTag = (tag: GauGANTag) => {
    setCurrentTag(tag)
    setCurrentColor(TAG_COLORS[tag].color)
  }

  const encodeGauGANTags = (): tf.Tensor2D => {
    const encoding = new Array(192).fill(0)
    const tags = Array.from(activePixels)
    // Ajout du tag de fond
    tags.push(selectedBackground)
    tags.forEach(tag => {
      if (Object.keys(TAG_COLORS).includes(tag)) {
        console.log(TAG_COLORS[tag])
        const index = Object.keys(TAG_COLORS).indexOf(tag)
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

    const output = gaugan.predict([tensor1, tensor2]) as tf.Tensor
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
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}>
        <div
          style={{
            marginTop: "30px",
            textAlign: "center",
            padding: "20px",
            display: "flex",
            justifyContent: "center",
            borderRadius: "10px",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
            margin: "20px auto",
            maxHeight: "300px",
          }}>
          <div
            style={{
              display: "inline-block",
              padding: "10px",
              backgroundColor: "gray",
            }}>
            {/* Color Palette */}
            <div
              style={{
                marginTop: "20px",
                textAlign: "center",
                display: "flex",
                flexWrap: "wrap",
                gap: "10px",
                justifyContent: "center",
                padding: "20px",
                borderRadius: "10px",
              }}>
              {Object.entries(TAG_COLORS).map(([tag, { color, label }]) => (
                <div
                  key={tag}
                  onClick={() => selectTag(tag as GauGANTag)}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: "5px",
                    cursor: "pointer",
                    color: "black",
                  }}>
                  <div
                    style={{
                      width: "30px",
                      height: "30px",
                      backgroundColor: color,
                      border:
                        currentTag === tag
                          ? "3px solid white"
                          : "1px solid #666",
                      borderRadius: "5px",
                    }}
                  />
                  <span
                    style={{
                      color: "white",
                      fontSize: "12px",
                      opacity: activePixels.has(tag as GauGANTag) ? 1 : 0.7,
                    }}>
                    {label}
                  </span>
                </div>
              ))}
            </div>
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
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              margin: "20px auto",
            }}>
            <button
              onClick={clearCanvas}
              style={{
                backgroundColor: "#555",
                color: "white",
                padding: "10px 20px",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                fontSize: "16px",
              }}>
              Clear Canvas
            </button>
          </div>
        </div>
        <div
          style={{
            display: "flex", // Utilisation de flexbox pour aligner horizontalement
            justifyContent: "space-evenly", // Espacement égal entre les éléments
            alignItems: "center", // Alignement vertical centré
            marginTop: "20px", // Espacement en haut
            gap: "20px", // Espace entre le canevas et l'image générée
          }}>
          {/* Canevas pour dessiner */}
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            style={{
              backgroundColor: TAG_COLORS[selectedBackground].color,
              borderRadius: "10px",
              width: "512px",
              height: "512px",
              imageRendering: "pixelated",
              cursor: "crosshair",
            }}
          />

          {/* Image générée */}
          {generatedImage && (
            <img
              src={generatedImage}
              alt="Generated"
              style={{
                width: "512px", // Dimensions identiques au canevas pour un alignement cohérent
                height: "512px",
                borderRadius: "10px",
                imageRendering: "pixelated",
              }}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export default App

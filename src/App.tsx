import {
  CSSProperties,
  SetStateAction,
  useEffect,
  useRef,
  useState,
} from "react"
import "./App.css"
import * as tf from "@tensorflow/tfjs"
import { ChromePicker } from "react-color" // Import du color picker

const gauganUrl =
  "https://huggingface.co/salim4n/gaugan-tfjs/resolve/main/model.json"

function App() {
  const [gaugan, setGaugaun] =
    useState<tf.GraphModel<string | tf.io.IOHandler>>()
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [color, setColor] = useState("#ffffff")
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [preHeating, setPreHeating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null) // État pour stocker l'image générée
  const [urlCss, setUrlCss] = useState<CSSProperties>({
    color: "lightblue",
    font: "small-caption",
    fontSize: "20px",
  }) // État pour stocker l'URL de la couleur sélectionnée
  const [heartCss, setHeartCss] = useState<CSSProperties>({})

  const startDrawing = (event: React.MouseEvent) => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!ctx) return

    setIsDrawing(true)
    ctx.strokeStyle = color // Utilisation de la couleur sélectionnée
    ctx.lineWidth = 5 // Épaisseur du trait
    ctx.lineJoin = "round" // Douceur des angles
    ctx.lineCap = "round" // Arrondi des extrémités des lignes
    ctx.beginPath()
    ctx.moveTo(event.nativeEvent.offsetX, event.nativeEvent.offsetY)
  }

  // Fonction pour arrêter le dessin
  const stopDrawing = () => {
    setIsDrawing(false)
  }

  // Fonction pour dessiner sur le canvas
  const draw = (event: React.MouseEvent) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!ctx) return

    ctx.lineTo(event.nativeEvent.offsetX, event.nativeEvent.offsetY)
    ctx.stroke() // Dessin de la ligne
  }

  // Fonction pour générer une image via GauGAN
  const generateImage = async () => {
    console.log("Generating image with GauGAN...")
    if (!canvasRef.current || !gaugan) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    console.log("canvas", canvas)
    console.log("ctx", ctx)
    if (!ctx) return

    // Récupérer l'image dessinée sous forme de tableau de données
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    console.log("imageData", imageData)

    // Convertir les pixels en tensor
    let inputTensor: tf.Tensor4D = tf.browser
      .fromPixels(imageData)
      .toFloat()
      .div(tf.scalar(255)) // Normalisation
    tf.print(inputTensor)

    // Obtenir les shapes d'entrée du modèle
    const inputShapes = gaugan.inputs.map(input => input.shape)
    console.log("Input shapes:", inputShapes) // Cela te donnera la forme attendue

    // Adapter l'entrée à la forme du modèle
    // Généralement, GauGAN prend des images de forme [1, height, width, channels]
    // const expectedShape = inputShapes[0] // Par exemple, ça pourrait être [1, 256, 256, 3]

    // Redimensionner si nécessaire
    // if (expectedShape) {
    //   if (
    //     inputTensor.shape[1] !== expectedShape[1] ||
    //     inputTensor.shape[2] !== expectedShape[2]
    //   ) {
    //     inputTensor = tf.image.resizeBilinear(inputTensor, [
    //       expectedShape[1],
    //       expectedShape[2],
    //     ])
    //   }
    // } else {
    //   console.error("Expected shape not found in model inputs")
    // }

    // Ajouter une dimension supplémentaire si le modèle attend [1, height, width, channels]
    inputTensor = inputTensor.expandDims(0) // Ajouter batch dimension si nécessaire
    tf.print(inputTensor)

    if (tf.isNaN(inputTensor).any().dataSync()[0]) {
      console.error("NaN detected in input tensor")
      return
    }
    // Générer une image avec GauGAN
    console.log("Generating image with GauGAN...")
    const output = (await gaugan.predict(inputTensor)) as tf.Tensor
    if (tf.isNaN(output).any().dataSync()[0]) {
      console.error("NaN detected in output tensor")
      return
    }
    tf.print(output)
    console.log("output", output)

    // Post-traitement de la sortie du modèle
    const imageTensor = output
      .squeeze()
      .mul(tf.scalar(255))
      .clipByValue(0, 255)
      .toInt() // Ramener les valeurs à l'échelle de 0 à 255
    tf.print(imageTensor)
    // Convertir la sortie en pixels
    const imageDataOutput = await tf.browser.toPixels(
      imageTensor as tf.Tensor3D
    )
    console.log(imageDataOutput)

    // Créer une nouvelle image à partir du résultat
    const newCanvas = document.createElement("canvas")
    newCanvas.width = canvas.width
    newCanvas.height = canvas.height
    const newCtx = newCanvas.getContext("2d")
    if (newCtx) {
      const newImageData = newCtx.createImageData(canvas.width, canvas.height)
      newImageData.data.set(imageDataOutput)
      newCtx.putImageData(newImageData, 0, 0)
      const generatedURL = newCanvas.toDataURL()
      setGeneratedImage(generatedURL) // Stocker l'image générée
    }
  }

  useEffect(() => {
    ;(async () => {
      tf.setBackend("webgl").then(() => console.log(tf.getBackend()))
      setLoading(true)
      const indexedDB = await window.indexedDB.databases()
      console.log(indexedDB)
      if (indexedDB.some((db: IDBDatabaseInfo) => db.name === "tensorflowjs")) {
        console.log("model already loaded from indexeddb")
        const model = await tf.loadGraphModel("indexeddb://gaugan-tfjs", {
          onProgress: progress => setProgress(progress),
        })
        setPreHeating(true)
        // Obtenir la forme d'entrée du modèle
        const inputShapes = model.inputs.map(input => input.shape)
        // [-1,192],
        // [-1,256,256,18]
        console.log("Input shapes:", inputShapes) // Afficher les formes d'entrée
        // Créer les tenseurs selon les formes d'entrée attendues
        const tensor1 = tf.zeros([1, 192]) // Modifie la forme en fonction des besoins
        const tensor2 = tf.zeros([1, 256, 256, 18]) // Modifie la forme en fonction des besoins
        // Vérifier que le modèle accepte bien les entrées sous forme de tableau
        const pred = await model.predict([tensor1, tensor2]) // Passer les tenseurs sous forme de tableau
        setPreHeating(false)
        console.log(pred)
        tf.dispose([tensor1, tensor2, pred]) // Libérer les tenseurs
        setGaugaun(model)
        setLoading(false) // Ajouté ici pour arrêter le chargement
        return
      } else {
        console.log("model not loaded from indexeddb")
        const model = await tf.loadGraphModel(gauganUrl, {
          onProgress: progress => setProgress(progress),
        })
        await model.save("indexeddb://gaugan-tfjs")
        setGaugaun(model)
      }
      console.log(gaugan)
      setLoading(false)
    })()
  }, [])

  if (preHeating) {
    return (
      <div>
        <h1 style={{ color: "white", textAlign: "center" }}>
          🌡️ pre-heating model...
        </h1>
      </div>
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
          <p style={{ color: "#888" }}>
            First loading can take a while... but don't worry, it's only once.
          </p>
          <p style={{ color: "#888" }}>
            Go take a coffee, this is going to be a while ☕
          </p>
        </div>
      )}

      {/* Section pour le color picker */}
      <div
        style={{
          marginTop: "30px",
          textAlign: "center",
          padding: "20px",
          borderRadius: "10px",
          boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
          maxWidth: "400px",
          margin: "20px auto",
        }}>
        <h3 style={{ marginBottom: "20px", color: "white" }}>
          Select a color for your sketch:
        </h3>
        <div
          style={{
            display: "inline-block",
            padding: "10px",
            borderRadius: "8px",
            backgroundColor: "#fff",
            cursor: `url(./assets/color-picker.svg), crosshair`,
          }}>
          <ChromePicker
            color={color}
            onChangeComplete={(newColor: { hex: SetStateAction<string> }) =>
              setColor(newColor.hex)
            }
            styles={{
              default: {
                picker: {
                  boxShadow: "0 2px 5px rgba(0,0,0,0.2)",
                  borderRadius: "10px",
                },
              },
            }}
          />
        </div>
      </div>

      {/* Canvas */}
      <div style={{ textAlign: "center", marginTop: "30px" }}>
        <canvas
          ref={canvasRef} // Référence pour le canvas
          id="canvas"
          width="800"
          height="600"
          onMouseDown={startDrawing} // Début du dessin
          onMouseMove={draw} // Mouvement de la souris pour dessiner
          onMouseUp={stopDrawing} // Fin du dessin
          onMouseLeave={stopDrawing} // Arrêter le dessin si la souris sort du canvas
          style={{
            backgroundColor: "white",
            borderRadius: "10px",
            cursor: "url('./assets/pen.svg') 4 12, crosshair",
          }}></canvas>
      </div>
      {/* Bouton pour générer l'image */}
      <div style={{ marginTop: "20px" }}>
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
          Generate Image with GauGAN
        </button>
      </div>
      {generatedImage && (
        <div style={{ marginTop: "30px" }}>
          <h3 style={{ color: "white" }}>Generated Image:</h3>
          <img
            src={generatedImage}
            alt="Generated"
            style={{
              maxWidth: "100%",
              borderRadius: "10px",
              boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
            }}
          />
        </div>
      )}
    </div>
  )
}

export default App

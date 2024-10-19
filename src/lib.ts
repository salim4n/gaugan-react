import { CANVAS_HEIGHT, CANVAS_WIDTH, GAUGAN_TAGS } from "./sd"
import * as tf from "@tensorflow/tfjs"

export const startDrawing = (
  event: React.MouseEvent,
  canvasRef: any,
  setIsDrawing: (arg0: boolean) => void,
  color: string
): void => {
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

export const stopDrawing = (setIsDrawing: (arg0: boolean) => void): void =>
  setIsDrawing(false)

export const draw = (
  event: React.MouseEvent,
  isDrawing: boolean,
  canvasRef: any
): void => {
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

export const encodeGauGANTags = (): tf.Tensor2D => {
  const encoding = new Array(192).fill(0)
  GAUGAN_TAGS.forEach((_, index) => {
    encoding[index] = 1
  })
  return tf.tensor2d([encoding], [1, 192])
}

export const generateImage = async (
  canvasRef: any,
  gaugan: any,
  setGeneratedImage: any
): Promise<void> => {
  if (!canvasRef.current || !gaugan) return

  const canvas = canvasRef.current
  const ctx = canvas.getContext("2d")
  if (!ctx) return

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  let inputTensor = tf.browser
    .fromPixels(imageData)
    .toFloat()
    .div(tf.scalar(255))

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
  const imageDataOutput = await tf.browser.toPixels(imageTensor as tf.Tensor3D)

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

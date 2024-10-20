import { useEffect, useState } from "react"
import * as tf from "@tensorflow/tfjs"
import { gauganUrl } from "./sd"

export default function useGaugan() {
  const [gaugan, setGaugan] =
    useState<tf.GraphModel<string | tf.io.IOHandler>>()
  const [progress, setProgress] = useState<number>(0)
  const [loading, setLoading] = useState<boolean>(false)
  const [preHeating, setPreHeating] = useState<boolean>(false)

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

  return {
    gaugan,
    progress,
    loading,
    preHeating,
  }
}

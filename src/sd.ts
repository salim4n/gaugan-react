// static data

export const gauganUrl =
  "https://huggingface.co/salim4n/gaugan-tfjs/resolve/main/model.json"

export const CANVAS_WIDTH = 256
export const CANVAS_HEIGHT = 256

export const TAG_COLORS = {
  cloud: { color: "#818181", label: "Cloud - Gray" },
  sky: { color: "#44ACCB", label: "Sky - Blue" },
  tree: { color: "#674A21", label: "Tree - Brown" },
  moutain: { color: "#5C785D", label: "Mountain - Green" },
  sea: { color: "#262C7E", label: "Sea - Blue" },
  grass: { color: "#149723", label: "Grass - Green" },
} as const

export type GauGANTag = keyof typeof TAG_COLORS

export const BACKGROUNDS_URL = ["/sky.jpg", "/grass.jpg"] as const

export type backgrounds = (typeof BACKGROUNDS_URL)[number]

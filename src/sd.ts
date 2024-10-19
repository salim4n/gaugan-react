// static data

export const gauganUrl =
  "https://huggingface.co/salim4n/gaugan-tfjs/resolve/main/model.json"

export const CANVAS_WIDTH = 256
export const CANVAS_HEIGHT = 256

// Définition des tags avec leurs couleurs associées
export const TAG_COLORS = {
  sky: { color: "#7CC8F8", label: "Sky - Light Blue" },
  cloud: { color: "#FFFFFF", label: "Cloud - White" },
  grass: { color: "#73B15B", label: "Grass - Green" },
  tree: { color: "#1B7334", label: "Tree - Dark Green" },
  mountain: { color: "#8B4513", label: "Mountain - Brown" },
  water: { color: "#2389DA", label: "Water - Blue" },
  earth: { color: "#815438", label: "Earth - Dark Brown" },
  road: { color: "#404040", label: "Road - Gray" },
  rock: { color: "#6B6B6B", label: "Rock - Dark Gray" },
  sand: { color: "#EFDC9E", label: "Sand - Light Brown" },
  snow: { color: "#FFFAFA", label: "Snow - Snow White" },
  building: { color: "#BC8F8F", label: "Building - Rosy Brown" },
  bush: { color: "#4A6741", label: "Bush - Forest Green" },
  flower: { color: "#FF69B4", label: "Flower - Pink" },
  sea: { color: "#006994", label: "Sea - Deep Blue" },
  river: { color: "#4682B4", label: "River - Steel Blue" },
  hill: { color: "#90A955", label: "Hill - Olive Green" },
  forest: { color: "#228B22", label: "Forest - Forest Green" },
} as const

export type GauGANTag = keyof typeof TAG_COLORS

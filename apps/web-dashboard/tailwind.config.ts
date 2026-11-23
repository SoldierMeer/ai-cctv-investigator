import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        command: {
          bg: "#040814",
          panel: "#0b1224",
          card: "#111b33",
          border: "#1e2d4d",
        },
      },
    },
  },
  plugins: [],
};

export default config;

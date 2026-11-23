import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI CCTV Investigator | Cross-Modal Surveillance Command Center",
  description:
    "Natural-language person retrieval across multi-camera CCTV feeds powered by CLIP ViT-B/16, IRRA 4-stripe alignment, and FAISS HNSW.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#040814] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-100">
        {children}
      </body>
    </html>
  );
}

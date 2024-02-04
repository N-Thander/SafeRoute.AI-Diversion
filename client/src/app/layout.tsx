import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SafeRoute.AI",
  description: "Your Road Companion",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <script
          src="https://apis.mappls.com/advancedmaps/api/89f90b894659bae06de91e13da508456/map_sdk?layer=vector&v=3.0&callback=initMap1"
          defer
          async
        ></script>
      </head>
      <body>
        <Navbar />
        {children}
      </body>
    </html>
  );
}

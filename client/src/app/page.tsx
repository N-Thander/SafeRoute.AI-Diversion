"use client";
import { TypeAnimation } from "react-type-animation";
import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="w-full mx-auto z-10 relative">
      <Image
        src="/assets/map.svg"
        alt="Background Image"
        layout="fill"
        objectFit="cover"
      />
      <div className="flex z-10 relative w-full flex-row md:w-[80%] items-center mx-auto justify-center h-[90vh] my-auto md:gap-20 flex-wrap-reverse">
        <div className="w-[90%] md:w-[40%] flex flex-col mx-auto items-center  md:items-start gap-6">
          <TypeAnimation
            sequence={[
              // Same substring at the start will only be typed out once, initially
              "Smoothing Roads, Smoothing Rides: Unveiling SafeRoutes",
              1000,
              "Your Intelligent Road Companion",
              1000,
            ]}
            wrapper="span"
            speed={200}
            style={{ display: "inline-block" }}
            className="font-bold text-2xl lg:text-4xl md:w-[300px] lg:w-[700px]  text-[#3b2476]"
            repeat={Infinity}
          />
          <p className="text-2xl font-semibold text-white">
            SafeRoute.AI - Your Smart Road Companion - Detect once alert the
            World ! Ensuring - that Every Identified Pothole Sends an Alert to
            All !
          </p>
          <Link href={"/dashboard"}>
            <button className="text-white text-xl px-10 py-3 bg-transparent font-semibold rounded-xl hover:bg-white hover:text-black border-2 border-white duration-300 transition-all">
              Get Started
            </button>
          </Link>
        </div>
        <div className="w-[90%] md:w-[40%] mx-auto">
          <img
            src="
          /assets/logo.jpeg"
            className="w-[80%] md:w-1/2 h-1/2 mx-auto object-cover rounded-full shadow-2xl"
            alt=""
          />
        </div>
      </div>

      {/* 
      <Map />
      <FileUpload /> */}
    </div>
  );
}

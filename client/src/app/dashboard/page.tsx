"use client";
import Image from "next/image";
import Link from "next/link";
import React from "react";
import { useRouter } from "next/navigation";
const cardData = [
  {
    title: "Explore",
    description: "Know all potholes in your area",
    image: "/assets/analyse.jpeg",
    buttonText: "Open Map",
    route: "/dashboard/map",
  },
  {
    title: "On The Go",
    image: "/assets/map.jpeg",
    description: "Know all potholes on your way to your destination",
    buttonText: "Analyse Now",
    route: "/dashboard/detect",
  },

  {
    title: "Saved Routes",
    image: "/assets/saved.jpeg",
    description: "Save your routes.",
    buttonText: "See Now",
    route: "/dashboard/saved-routes",
  },
];
interface RouteCardProps {
  title: string;
  image: string;
  buttonText: string;
  route: string;
}

const RouteCard = ({ title, image, route, buttonText }: RouteCardProps) => {
  const router = useRouter();
  return (
    <div className="bg-white px-10 w-[90%] md:w-[400px] md:h-[500px] py-5 border-blue-600 border-2 rounded-xl flex flex-col items-center justify-center gap-5">
      <img
        src={image}
        className="w-[220px] h-[300px]"
        height={0}
        width={0}
        alt=""
      />
      <h1 className="text-xl font-semibold text-center text-black tracking-widest">
        {title}
      </h1>

      <Link
        href={route}
        className="px-10 py-2 text-md font-semibold hover:bg-opacity-50 bg-violet-700 text-white tracking-wider rounded-xl"
        onClick={() => {
          router.push(route);
        }}
      >
        {buttonText}
      </Link>
    </div>
  );
};

const page = () => {
  return (
    <div className="relative h-full flex flex-col md:min-h-screen fill-transparent">
      {" "}
      <h1 className="font-semibold text-3xl my-5 text-white text-center w-full">
        Features
      </h1>
      <div className="flex flex-row gap-10 items-center justify-center mx-auto flex-wrap">
        {cardData.map((data, index) => {
          return (
            <RouteCard
              key={index}
              title={data.title}
              image={data.image}
              buttonText={data.buttonText}
              route={data.route}
            />
          );
        })}
      </div>
    </div>
  );
};

export default page;

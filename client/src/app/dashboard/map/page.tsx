"use client";
import Map from "@/components/Map";
import TableData from "@/components/Table";
import axios from "axios";
import React, { useEffect } from "react";

const page = () => {
  const [potholes, setPotholes] = React.useState([]);
  useEffect(() => {
    const getPotholes = async () => {
      try {
        const res = await axios.get("http://localhost:5000/getpotholes");
        setPotholes(res.data);
      } catch (error) {
        console.error("Error fetching potholes:", error);
      }
    };
    getPotholes();
  }, []);
  return (
    <div className="flex flex-col items-center  justify-center gap-5 w-full mx-auto mt-[20px] mb-[50px]">
      <h1 className="text-2xl font-semibold tracking-wide text-white ">
        Beware of your nearby Potholes
      </h1>
      <div className="flex flex-col md:flex-row flex-wrap items-center gap-5 w-[80%]  md:w-[90%]">
        <Map potholes={potholes} />
        <TableData potholes={potholes} />
      </div>
    </div>
  );
};

export default page;

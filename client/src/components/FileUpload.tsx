"use client";
import axios from "axios";
import { FileInput, Label } from "flowbite-react";
import { useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";
import { Spinner } from "flowbite-react";

function FileUpload() {
  const [location, setLocation] = useState<any>({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const [videoURL, setVideoURL] = useState<string | null>(null);
  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(successCallback, errorCallback, {
        enableHighAccuracy: true,
        timeout: 5000,
        maximumAge: 0,
      });
    } else {
      // Geolocation is not supported
      setError("Geolocation is not supported by your browser.");
    }
  }, []); // Empty dependency array to run the effect only once

  // Success callback function
  const successCallback = (position: any) => {
    const latitude = position.coords.latitude;
    const longitude = position.coords.longitude;

    // Display the location
    setLocation({ latitude, longitude });
  };

  // Error callback function
  const errorCallback = (error: any) => {
    switch (error.code) {
      case error.PERMISSION_DENIED:
        setError("User denied the request for Geolocation.");
        break;
      case error.POSITION_UNAVAILABLE:
        setError("Location information is unavailable.");
        break;
      case error.TIMEOUT:
        setError("The request to get user location timed out.");
        break;
      case error.UNKNOWN_ERROR:
        setError("An unknown error occurred.");
        break;
      default:
        setError("An error occurred while fetching the location.");
    }
  };
  const [base64Video, setBase64Video] = useState<any>("");

  const handleFileChange = (e: any) => {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.readAsDataURL(file);

    reader.onloadend = () => {
      setVideoURL(reader.result as string);
      setBase64Video(reader.result);
    };

    console.log("File uploaded:", file);
  };

  const handleSubmit = async () => {
    try {
      await axios.post(
        "http://localhost:5000/upload",
        { video: base64Video, location: location },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      console.log("Video uploaded successfully!");
      setLoading(true);
      setTimeout(() => {
        router.push("/dashboard/map");
        setVideoURL(null);
        setLoading(false);
      }, 5000);
    } catch (error) {
      console.error("Error uploading video:", error);
    }
  };

  return (
    <div className=" gap-10  items-center justify-center flex flex-col mx-auto">
      <h1 className="text-2xl  text-white font-semibold tracking-wider ">
        Upload a Road Footage to Analyse the Potholes and get markers in the map
      </h1>
      <Label
        htmlFor="dropzone-file"
        className="dark:hover:bg-bray-800 w-[90%] md:w-1/3 flex h-64  cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-blue-300 bg-gray-300 hover:bg-gray-200 dark:border-gray-600 dark:bg-gray-700 dark:hover:border-gray-500 dark:hover:bg-gray-600"
      >
        <div className="flex flex-col items-center justify-center pb-6 pt-5">
          <svg
            className="mb-4 h-8 w-8 text-gray-500 dark:text-gray-400"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 20 16"
          >
            <path
              stroke="currentColor"
              strokeLinecap="round"
              strokeWidth="2"
              d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
            />
          </svg>
          <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
            <span className="font-semibold">Click to upload</span> or drag and
            drop
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            SVG, PNG, JPG or GIF (MAX. 800x400px)
          </p>
        </div>
        <FileInput
          id="dropzone-file"
          className="hidden"
          accept="video/mp4/webp"
          onChange={handleFileChange}
        />
      </Label>
      {videoURL && (
        <video
          controls
          width="400"
          className="mt-4"
          // Assuming videoURL is a valid data URL
          src={videoURL}
          typeof="video/mp4/webp"
        >
          Your browser does not support the video tag.
        </video>
      )}
      <button
        onClick={() => {
          handleSubmit();
        }}
        className="bg-blue-900 hover:opacity-50 text-white text-lg text-center px-6 py-1 font-semibold rounded-xl"
      >
        {loading ? <Spinner /> : "Submit"}
      </button>
    </div>
  );
}

export default FileUpload;

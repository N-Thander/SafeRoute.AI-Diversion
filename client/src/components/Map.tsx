"use client";

import { Spinner } from "flowbite-react";
import { mappls } from "mappls-web-maps";
import { useEffect, useRef, useState } from "react";

interface GeoDataProps {
  type: string;
  features: Array<{
    type: string;
    properties: {
      htmlPopup: string;
    };
    geometry: {
      type: string;
      coordinates: [number, number];
    };
  }>;
}

interface potholeProps {
  potholes: Array<any>;
}

const Map = ({ potholes }: potholeProps) => {
  const [loading, setLoading] = useState(true);
  const markerArray: any = [];
  const mapping = async () => {
    potholes.forEach((item) => {
      markerArray.push({
        type: "Feature",
        properties: {
          htmlPopup: item.address,
        },
        geometry: {
          type: "Point",
          coordinates: [item.latitude, item.longitude],
        },
      });
    });
  };
  useEffect(() => {
    const getLocation = async () => {
      try {
        const position = await new Promise<GeolocationPosition>(
          (resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject);
          }
        );

        const { latitude, longitude } = position.coords;
        await mapping();
        initMap1(latitude, longitude);
        setLoading(false);
      } catch (error: any) {
        console.error("Error getting location:", error.message);
      }
    };
    const initMap1 = (latitude: any, longitude: any) => {
      console.log("Initializing map with coordinates:", latitude, longitude);
      let map: any;

      if (window.mappls && window.mappls.Map) {
        map = new window.mappls.Map("map", {
          center: [latitude, longitude],
          zoomControl: true,
          location: true,
        });

        if (map && typeof map.addListener === "function") {
          map.addListener("load", function () {
            var geoData = {
              type: "FeatureCollection",
              features: markerArray,
            };

            var marker = window.mappls.Marker({
              map: map,
              position: geoData,
              icon_url: "https://apis.mapmyindia.com/map_v3/1.png",
              fitbounds: true,
              clusters: true,
              clustersIcon: "https://mappls.com/images/2.png",
              fitboundOptions: {
                padding: 120,
                duration: 1000,
              },
              popupOptions: {
                offset: { bottom: [0, -20] },
              },
            });
          });
        }
      }
    };

    getLocation();
  }, [potholes]);

  return (
    <>
      <div
        id="map"
        className=" flex flex-col items-center justify-center w-[60%] mx-auto h-[80vh]"
      >
        {loading && <Spinner color={"#000000"} />}
      </div>
    </>
  );
};
export default Map;

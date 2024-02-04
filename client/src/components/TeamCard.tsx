"use client";

import Image from "next/image";
import React from "react";
import { CardBody, CardContainer, CardItem } from "./ui/3d-card";
import { BsGithub, BsInstagram, BsLinkedin } from "react-icons/bs";

export function ThreeDCardDemo({
  name,
  role,
  linkedin,
  github,
  instagram,
  image,
}: {
  name: string;
  role: string;
  linkedin: string;
  github: string;
  instagram: string;
  image: string;
}) {
  return (
    <CardContainer className="inter-var ">
      <CardBody className="bg-gray-50 relative group/card  dark:hover:shadow-2xl dark:hover:shadow-emerald-500/[0.1] dark:bg-black dark:border-white/[0.2] border-black/[0.1] w-auto sm:w-[250px] h-full rounded-xl p-6 border  ">
        <div className="flex flex-col items-center justify-center">
          <CardItem translateZ="100" className="w-full  mx-auto">
            <Image
              src={image}
              height="900"
              width="1000"
              className="h-[200px] w-full object-cover rounded-xl group-hover/card:shadow-xl"
              alt="thumbnail"
            />
            <h1 className="text-xl text-center font-bold dark:text-white mt-10">
              {name}
            </h1>
            <h1 className="text-lg text-center font-bold dark:text-white mt-2">
              {role}
            </h1>
            <CardItem
              translateZ={20}
              as={"text"}
              className="flex justify-center w-full gap-3 items-center mt-5"
            >
              <a
                href={instagram}
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2 rounded-full bg-[#6a45e4] font-bold text-xl text-white hover:bg-white hover:text-[#6a45e4] "
              >
                <BsInstagram className="h-[25px] w-[25px]" />
              </a>
              <a
                href={github}
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2 rounded-full bg-[#0c0b0f] font-bold text-xl text-white hover:bg-white hover:text-[#0c0b0f]"
              >
                <BsGithub className="h-[25px] w-[25px]" />
              </a>
              <a
                href={linkedin}
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2 rounded-full bg-[#215594] font-bold text-xl text-white hover:bg-white hover:text-[#215594]"
              >
                <BsLinkedin className="h-[25px] w-[25px]" />
              </a>
            </CardItem>
          </CardItem>

          {/* <CardItem
            translateZ={20}
            as="button"
            className="px-4 py-2 rounded-xl bg-black dark:bg-white dark:text-black text-white text-xs font-bold"
          >
            Sign up
          </CardItem> */}
        </div>
      </CardBody>
    </CardContainer>
  );
}

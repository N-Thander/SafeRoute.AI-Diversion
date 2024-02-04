import { ThreeDCardDemo } from "@/components/TeamCard";
import React from "react";

const teamData = [
  {
    name: "Arghya Dutta",
    role: "Team Lead , Full Stack Developer",
    image:
      "https://media.licdn.com/dms/image/D4D03AQEXkmbOuqFchw/profile-displayphoto-shrink_800_800/0/1698006488192?e=1712188800&v=beta&t=lSTns7b_B-iJywZuG4-0xIszqh-VRxnhX86UMWzC2-4",
    linkedin: "https://www.linkedin.com/in/arghya-dutta-aa8b3b1b2/",
    github: "",
    instagram: "",
  },
  {
    name: "Soumyajit Biswas",
    role: "Data Scientist",
    image:
      "https://media.licdn.com/dms/image/D4D03AQHBsT6XIjAsSQ/profile-displayphoto-shrink_400_400/0/1697205381955?e=1712188800&v=beta&t=6sd51XaIsygwGFbRVc2sIVQiN7Zm7j9oB90omZNtI70",
    linkedin: "https://www.linkedin.com/in/soumyajit-biswas-909a671a0/",
    github: "",
    instagram: "",
  },

  {
    name: "Nasiruddin Thander",
    role: "ML Engineer",
    image: "https://i.postimg.cc/tCVXryVH/nasir.jpg",
    linkedin:
      "https://www.linkedin.com/in/https://www.linkedin.com/in/nasiruddin-thander-356596284//",
    github: "",
    instagram: "",
  },
  {
    name: "Soumyaraj Bag",
    role: "Web-App Developer",
    image:
      "https://media.licdn.com/dms/image/D5635AQG2v4L6unLGQA/profile-framedphoto-shrink_400_400/0/1692910223871?e=1707246000&v=beta&t=t70FX3lb6xPIG0ipnsJ3auIrKx3ZbmMJfBcTu8SxGiY",
    linkedin: "https://www.linkedin.com/in/soumyarajbag/",
    github: "",
    instagram: "",
  },
];
const page = () => {
  return (
    <div className="my-10 md:max-w-[80%] mx-auto flex flex-col items-center">
      <h1 className="text-4xl text-white font-semibold tracking-wider ">
        Team :{" "}
        <span className="hover:text-orange-700 cursor-pointer">
          DataDynamos
        </span>{" "}
      </h1>
      <div className="flex flex-wrap items-center md:gap-10 justify-center mx-auto ">
        {teamData.map((item, index) => {
          return (
            <ThreeDCardDemo
              key={index}
              name={item.name}
              image={item.image}
              role={item.role}
              linkedin={item.linkedin}
              github={item.github}
              instagram={item.instagram}
            />
          );
        })}
      </div>
    </div>
  );
};

export default page;

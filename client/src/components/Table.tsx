"use client";
import { useMemo, useState } from "react";
import { Table } from "flowbite-react";

export default function TableData({
  potholes,
}: {
  potholes: Array<any>;
}): JSX.Element {
  console.log(potholes);
  const [loading, setLoading] = useState(true);
  const memoizedTableRows = useMemo(() => {
    return potholes.map((pothole: any, index: number) => (
      <Table.Row
        key={index}
        className="bg-white dark:border-gray-700 dark:bg-gray-800"
      >
        <Table.Cell className="whitespace-nowrap font-medium text-gray-900 dark:text-white">
          {pothole.address.split(",")[5]}
        </Table.Cell>
        <Table.Cell className="text-black">{pothole.longitude}</Table.Cell>
        <Table.Cell className="text-black">{pothole.latitude}</Table.Cell>
      </Table.Row>
    ));
  }, [potholes]);

  setTimeout(() => {
    setLoading(false);
  }, 3000);
  return (
    <div className="flex flex-col items-center gap-2">
      <h1 className="text-white text-lg font-semibold">
        {" "}
        Co-ordinates of Potholes
      </h1>
      <div className="overflow-scroll">
        <Table>
          <Table.Head className="text-black text-md bg-gray-500 rounded-xl border-1 px-5 py-2">
            <Table.HeadCell>Location</Table.HeadCell>
            <Table.HeadCell className="text-black">Longitude</Table.HeadCell>
            <Table.HeadCell className="text-black">Latitude</Table.HeadCell>
          </Table.Head>
          <Table.Body className="divide-y">
            {!loading && memoizedTableRows}
          </Table.Body>
        </Table>
      </div>
    </div>
  );
}

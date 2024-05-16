"use client";
import { Chatbot } from "@/components/chatbot";
import { GoogleMaps } from "@/components/google-maps";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState } from "react";

export default function Dashboard() {
    const [responses, setResponses] = useState<any>([]);

    return (
        <div className="py-10">
            <Tabs defaultValue="map" className="w-10/12 mx-auto">
                <TabsList className="grid w-full grid-cols-2 rounded-full">
                    <TabsTrigger value="map" className="rounded-full">Map of Benin</TabsTrigger>
                    <TabsTrigger value="chatbot" className="rounded-full">Chatbot</TabsTrigger>
                </TabsList>
                <TabsContent value="map">
                    <GoogleMaps />
                </TabsContent>
                <TabsContent value="chatbot">
                    <Chatbot responses={responses} setResponses={setResponses} />
                </TabsContent>
            </Tabs>
        </div>
    )
}
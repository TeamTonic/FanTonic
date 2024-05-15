import { Chatbot } from "@/components/chatbot";
import { GoogleMaps } from "@/components/google-maps";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Dashboard() {
    return (
        <div className="py-10">
            <Tabs defaultValue="map" className="w-11/12 mx-auto">
                <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="map">Map of Benin</TabsTrigger>
                    <TabsTrigger value="chatbot">Chatbot</TabsTrigger>
                </TabsList>
                <TabsContent value="map">
                    <GoogleMaps />
                </TabsContent>
                <TabsContent value="chatbot">
                    <Chatbot />
                </TabsContent>
            </Tabs>
        </div>
    )
}
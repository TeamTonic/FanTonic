import { GoogleMaps } from "@/components/google-maps";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Dashboard() {
    return (
        <div className="py-6">
            <Tabs defaultValue="map" className="w-10/12 mx-auto">
                <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="map">Map of Benin</TabsTrigger>
                    <TabsTrigger value="chatbot">Chatbot</TabsTrigger>
                </TabsList>
                <TabsContent value="map">
                    <GoogleMaps />
                </TabsContent>
                <TabsContent value="chatbot">
                    This is the Chatbot
                </TabsContent>
            </Tabs>
        </div>
    )
}
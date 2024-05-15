"use client";
import { Loader } from "@googlemaps/js-api-loader";
import { Loader2 } from "lucide-react";
import { useEffect, useRef } from "react";

export const GoogleMaps = () => {
    const mapRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const initializeMap = async () => {
            const loader = new Loader({
                apiKey: process.env.NEXT_PUBLIC_MAPS_API_KEY as string,
                version: "quartely",
            });

            const { Map } = await loader.importLibrary("maps");

            const locationOfBenin = {
                lat: 9.30769,
                lng: 2.315834,
            }

            const options: google.maps.MapOptions = {
                center: locationOfBenin,
                zoom: 15,
                mapId: "NEXT_MAP_ID"
            };

            const map = new Map(mapRef.current as HTMLDivElement, options);
        }
        initializeMap();
    }, [])

    return (
        <div className="h-[700px]" ref={mapRef}><Loader2 size={37} className="animate-spin" /></div>
    )
}
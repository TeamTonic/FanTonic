import Image from "next/image";
import { ModeToggle } from "./mode-toggle";
import { Button } from "./ui/button";

export const Header = () => {
    return (
        <header className="flex justify-between items-center w-full px-[5vw] border-b py-1.5">
            <div className="flex items-center cursor-pointer gap-x-1.5 hover:opacity-85">
                <Image src={"/logo.png"} width={80} height={80} alt="Logo" />
                <div className="space-x-1.5 max-sm:hidden">
                    <span className="text-3xl font-black">Zang</span>
                    <span className="text-3xl font-black text-primary/90">beto</span>
                </div>
            </div>
            <div className="flex gap-x-5">
                <ModeToggle />
                <Button className="text-base">Prompt in Fon now!</Button>
            </div>
        </header>
    )
}
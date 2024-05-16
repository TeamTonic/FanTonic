import { quick_answers } from "@/lib/constants";
import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowUp, Mic, X } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { ReactMic } from "react-mic";
import { z } from "zod";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { DropdownMenu, DropdownMenuContent, DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuSeparator, DropdownMenuTrigger } from "./ui/dropdown-menu";
import { Form, FormControl, FormField, FormItem, FormMessage } from "./ui/form";
import { Input } from "./ui/input";

const formSchema = z.object({
    prompt: z.string().min(10, {
        message: "Minimum length of your question should be 10 characters."
    })
});


export const Chatbot = ({ responses, setResponses }: any) => {
    const [voice, setVoice] = useState(false);
    const [language, setLanguage] = useState<string>("English");

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            prompt: "",
        },
    });

    async function onSubmit(values: z.infer<typeof formSchema>) {
        try {
            const response = await fetch("http://127.0.0.1:5000/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ "query_string": values.prompt }),
            });

            const data = await response.json();

            setResponses((prevResponses: any) => [
                ...prevResponses,
                { question: values.prompt },
                { answer: data.text },
            ]);

            form.reset();
        } catch (err: any) {
            throw new Error(err.message);
        }
    }

    const onStop = async (blob) => {
        try {
            const formData = new FormData();
            formData.append('audioBlob', blob, 'audio.wav'); // Append the blob with a filename
    
            const response = await fetch("http://127.0.0.1:5000/tts", {
                method: "POST",
                body: formData,
            });
    
            const data = await response.json();
    
            setResponses((prevResponses) => [
                ...prevResponses,
                { question: data.question },
                { answer: data.text },
            ]);
        } catch (err) {
            console.error("Error:", err);
        }
    }
    

    const startHandle = () => {
        setVoice(true);
    };

    const endHandle = () => {
        setVoice(false);
    };

    return (
        <section className="flex flex-col justify-between py-10 min-h-[720px]">
            {responses.length > 0 ?
                <div className="flex flex-col gap-y-3">
                    {responses.map((response: any, index: any) => (
                        <div key={index}>
                            {
                                response.question &&
                                <div className="px-5 py-4 mt-3 ml-auto rounded-full bg-secondary">{response.question}</div>
                            }
                            {
                                response.answer &&
                                <div className="px-5 py-4 rounded-full">
                                    {response.answer}
                                </div>
                            }
                        </div>
                    ))}
                </div>
                : <div className="flex flex-wrap gap-6 pb-16 max-md:justify-center">
                    {quick_answers.map((answer, index) => (
                        <Card
                            key={index}
                            className="w-[300px] h-[170px] bg-card p-4 hover:bg-secondary cursor-pointer duration-150"
                            onClick={() => onSubmit({ prompt: answer.question })}
                        >
                            <div className="flex items-center text-xl gap-x-3">
                                <answer.icon size={35} />
                                {answer.topic}
                            </div>
                            <CardContent className="mt-5">
                                {answer.question}
                            </CardContent>
                        </Card>
                    ))}
                </div>
            }

            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="relative mt-11">
                    <FormField
                        control={form.control}
                        name="prompt"
                        render={({ field }) => (
                            <FormItem className="flex flex-col">
                                <FormControl>
                                    <Input
                                        placeholder="What language do people speek in south Benin?"
                                        className="py-8 pl-6 pr-10 text-lg rounded-full bg-secondary"
                                        {...field}
                                    />
                                </FormControl>
                                <FormMessage className="pl-4 text-base" />
                            </FormItem>
                        )}
                    />
                    <ReactMic className="w-0" record={voice} onStop={onStop} mimeType="audio/wav" />
                    <div className="flex justify-end gap-4 mt-4">
                        <Button type="submit" disabled={!form.getValues("prompt")} className="p-3 py-6 rounded-full">
                            <ArrowUp />
                        </Button>
                        {
                            !voice ?
                                <Button
                                    type="button"
                                    onClick={() => startHandle()}
                                    className="p-3 py-6 rounded-full">
                                    <Mic />
                                </Button>
                                :
                                <Button
                                    type="button"
                                    variant={"destructive"}
                                    onClick={() => endHandle()} className="p-3 py-6 rounded-full">
                                    <X />
                                </Button>
                        }
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button variant={"secondary"} className="p-3 py-6 text-base rounded-full">{language}</Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent className="w-56">
                                <DropdownMenuLabel>Pick a language for Speech</DropdownMenuLabel>
                                <DropdownMenuSeparator />
                                <DropdownMenuRadioGroup defaultValue={"English"} value={language} onValueChange={setLanguage}>
                                    <DropdownMenuRadioItem value="English">English</DropdownMenuRadioItem>
                                    <DropdownMenuRadioItem value="French">French</DropdownMenuRadioItem>
                                    <DropdownMenuRadioItem value="Fongbe">Fongbe</DropdownMenuRadioItem>
                                </DropdownMenuRadioGroup>
                            </DropdownMenuContent>
                        </DropdownMenu>
                    </div>
                </form>
            </Form>
        </section >
    )
}
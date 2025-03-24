import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function AboutPage() {
  return (
    <div className="flex flex-col items-center min-h-[calc(100vh-4rem)] py-24 px-4">
      <div className="w-full max-w-[980px]">
        <h1 className="text-4xl font-bold mb-12 text-center bg-clip-text text-transparent bg-gradient-to-b from-foreground to-foreground/70">
          About UFC Stats
        </h1>
        <div className="grid gap-8 md:grid-cols-2">
          <Card className="bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Our Mission</CardTitle>
              <CardDescription>
                Providing accurate statistics and predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                We aim to provide MMA fans and analysts with comprehensive fighter statistics
                and data-driven fight predictions using advanced analytics and machine learning.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>How It Works</CardTitle>
              <CardDescription>
                The technology behind our predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Our prediction system analyzes historical fight data, fighter statistics,
                and various other metrics to generate accurate fight outcome predictions.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
} 